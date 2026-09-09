#pragma once
#include <pybind11/stl.h>
// For next-open structural signals only. Markets are [open_time_s,O,H,L,C].
// Valuations use completed closes or contemporaneous opens, never future H/L/C.
py::dict nav_account(D data,D tests,std::vector<D> markets,double risk,double begin,double end) {
    if(markets.size()!=4)throw std::runtime_error("Four market arrays required");
    auto a=data.unchecked<2>();auto q=tests.unchecked<2>();
    const double* m[4];int64_t len[4],ptr[4];double mid[4]={0,0,0,0},stamp[4]={-1,-1,-1,-1};
    for(int k=0;k<4;k++) {
        if(markets[k].ndim()!=2||markets[k].shape(1)!=5)throw std::runtime_error("Market shape must be Nx5");
        m[k]=markets[k].data();len[k]=markets[k].shape(0);ptr[k]=0;
        // No future close at the left boundary; completed bars only.
        while(ptr[k]<len[k]&&m[k][ptr[k]*5]+60<=begin){mid[k]=m[k][ptr[k]*5+4];stamp[k]=m[k][ptr[k]*5]+60;ptr[k]++;}
    }
    struct Pos {int64_t ix=-1;int inst=0;double units=0,entry=0,stop=0,exit=0,r=0,entry_nav=0;};
    Pos pos[2];double bal[2]={100,100};double peak=200,dd=0,closed_peak=200,closed_dd=0,gain=0,loss=0;
    double first_nav_million=-1,first_closed_million=-1;
    double min_size_ratio=1,max_size_ratio=0,max_leverage=0;int rejected=0,stale=0,wins=0,forced=0;
    std::vector<int64_t> accepted;std::vector<double> close_times,close_pnl,nav_t,nav_values;
    auto pip=[](int k){return k==3?.01:.0001;};
    auto conversion=[&](int inst,double quote_pnl) {
        if(inst==2)return quote_pnl*(mid[1]+(quote_pnl<0?.00008:-.00008));
        if(inst==3)return quote_pnl/(mid[3]+(quote_pnl<0?-.008:.008));
        return quote_pnl;
    };
    auto unreal=[&](int side) {
        const auto& p=pos[side];if(p.ix<0)return 0.;int sign=side?1:-1;
        double closeprice=mid[p.inst]-sign*.8*pip(p.inst);
        return conversion(p.inst,p.units*sign*(closeprice-p.entry));
    };
    auto record_nav=[&](double time,bool save) {
        double value=bal[0]+bal[1]+unreal(0)+unreal(1);peak=std::max(peak,value);
        if(value>=1000000.&&first_nav_million<0)first_nav_million=time;
        dd=std::min(dd,(value-peak)/peak);
        if(save){nav_t.push_back(time);nav_values.push_back(value);}
        return value;
    };
    auto close=[&](int side,double time,bool force) {
        auto& p=pos[side];if(p.ix<0)return;
        double pnl=force?unreal(side):conversion(p.inst,p.units*p.stop*p.r);
        bal[side]+=pnl;if(pnl>0){wins++;gain+=pnl;}else loss-=pnl;
        close_times.push_back(time);close_pnl.push_back(pnl);if(force)forced++;
        p.ix=-1;double sum=bal[0]+bal[1];closed_peak=std::max(closed_peak,sum);
        if(sum>=1000000.&&first_closed_million<0)first_closed_million=time;
        closed_dd=std::min(closed_dd,(sum-closed_peak)/closed_peak);
    };
    int64_t ti=0;while(ti<a.shape(0)&&a(ti,0)<begin)ti++;
    {py::gil_scoped_release unlock;
    double last=begin;int64_t iterations=0;
    while(true) {
        double now=end;
        for(int k=0;k<4;k++)if(ptr[k]<len[k])now=std::min(now,m[k][ptr[k]*5]);
        if(now>=end)break;
        if(now<begin) { // This can only be a raw bar straddling a non-minute boundary.
            for(int k=0;k<4;k++)if(ptr[k]<len[k]&&m[k][ptr[k]*5]==now)ptr[k]++;
            continue;
        }
        // Previous completed bars are visible at this time, including FX conversion.
        for(int k=0;k<4;k++)if(ptr[k]>0) {
            int64_t p=ptr[k]-1;
            if(m[k][p*5]+60<=now){mid[k]=m[k][p*5+4];stamp[k]=m[k][p*5]+60;}
        }
        for(int s=0;s<2;s++)if(pos[s].ix>=0&&pos[s].exit<=now)close(s,pos[s].exit,false);
        record_nav(now,false);
        // Current OPEN prices are knowable; current high/low/close are never read.
        for(int k=0;k<4;k++)if(ptr[k]<len[k]&&m[k][ptr[k]*5]==now){mid[k]=m[k][ptr[k]*5+1];stamp[k]=now;}
        bool changed=false;
        while(ti<a.shape(0)&&a(ti,0)<=now) {
            int64_t i=ti++;if(a(i,0)!=now)continue;
            bool keep=true;for(int64_t j=0;j<q.shape(0);j++) {
                double value=a(i,int(q(j,0)));if(!std::isfinite(value)||q(j,1)*(value-q(j,2))<0){keep=false;break;}}
            if(!keep)continue;
            int side=int(a(i,3)),inst=int(a(i,2));if(pos[side].ix>=0)continue;
            if(mid[inst]<=0||(inst==2&&(mid[1]<=0||now-stamp[1]>300))){stale++;continue;}
            double nav[2]={bal[0]+unreal(0),bal[1]+unreal(1)};
            double base=std::min(nav[0],nav[1]);if(base<=0)continue;
            double entry=a(i,4),stop=a(i,5);
            double perunit=inst==3?stop/mid[3]:(inst==2?stop*mid[1]:stop);
            double notional=inst==3?1.:(inst==2?entry*mid[1]:entry);
            // Broker-sized integer units, capped by the actual smaller NAV.
            double units=std::floor(std::min(base*risk/perunit,base*50/notional));
            if(units<1){rejected++;continue;}
            double margin=units*notional/50;
            if(margin>nav[side]+1e-10){rejected++;continue;}
            double ratio=units*perunit/base;min_size_ratio=std::min(min_size_ratio,ratio);max_size_ratio=std::max(max_size_ratio,ratio);
            max_leverage=std::max(max_leverage,units*notional/nav[side]);
            pos[side]={i,inst,units,entry,stop,a(i,1),a(i,6),nav[side]};accepted.push_back(i);changed=true;
        }
        record_nav(now,changed||iterations%1440==0);
        for(int k=0;k<4;k++)if(ptr[k]<len[k]&&m[k][ptr[k]*5]==now)ptr[k]++;
        last=now;iterations++;
    }
    // Incorporate the final completed minute, but not a candle closing after end.
    for(int k=0;k<4;k++)if(ptr[k]>0) {
        int64_t p=ptr[k]-1;if(m[k][p*5]+60<=end){mid[k]=m[k][p*5+4];stamp[k]=m[k][p*5]+60;}}
    for(int s=0;s<2;s++)if(pos[s].ix>=0)close(s,std::min(pos[s].exit,end),pos[s].exit>end);
    record_nav(end,true);}
    py::dict out;double years=(end-begin)/(86400*365.25);
    out["n"]=accepted.size();out["wr"]=accepted.empty()?0.:100.*wins/accepted.size();out["pf_usd"]=loss?gain/loss:0;
    out["nav_dd"]=dd*100;out["closed_dd"]=closed_dd*100;out["equity"]=bal[0]+bal[1];
    out["buy_equity"]=bal[1];out["sell_equity"]=bal[0];out["tpy"]=accepted.size()/years;
    out["log_growth"]=std::log(std::max((bal[0]+bal[1])/200.,1e-100))/years;
    out["forced_boundary_exits"]=forced;out["rejected"]=rejected;out["stale_conversion_rejects"]=stale;
    out["max_initial_leverage"]=max_leverage;out["max_effective_risk_fraction"]=max_size_ratio;
    out["observed_nav_days_to_1m"]=first_nav_million<0?-1.:(first_nav_million-begin)/86400;
    out["observed_closed_days_to_1m"]=first_closed_million<0?-1.:(first_closed_million-begin)/86400;
    out["accepted"]=arr(accepted);out["close_times"]=arr(close_times);out["close_pnl"]=arr(close_pnl);
    out["nav_times"]=arr(nav_t);out["nav_values"]=arr(nav_values);return out;
}
