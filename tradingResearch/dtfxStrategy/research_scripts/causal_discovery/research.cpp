#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <array>
#include "zone_detect.hpp"
#include "backtest.hpp"
namespace py = pybind11;
using D = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
template<class T> py::array_t<T> arr(const std::vector<T>& v) {
    py::array_t<T> a(v.size()); std::copy(v.begin(),v.end(),a.mutable_data()); return a;
}
py::tuple aggregate(I t,D op,D hi,D lo,D cl,int minutes) {
    std::vector<int64_t> times;std::vector<double> o,h,l,c;
    int64_t width=int64_t(minutes)*60*1000000000LL;
    for(int64_t i=0;i<t.size();) {
        int64_t bucket=t.data()[i]/width*width,j=i+1;
        double high=hi.data()[i],low=lo.data()[i];
        while(j<t.size()&&t.data()[j]<bucket+width){high=std::max(high,hi.data()[j]);low=std::min(low,lo.data()[j]);j++;}
        // Exclude partial first/last buckets. Internal missing minutes are absent
        // observations, never backfilled from future prices.
        if(bucket>=t.data()[0]&&bucket+width<=t.data()[t.size()-1]+60000000000LL){
            times.push_back(bucket);o.push_back(op.data()[i]);h.push_back(high);l.push_back(low);c.push_back(cl.data()[j-1]);}
        i=j;
    }
    return py::make_tuple(arr(times),arr(o),arr(h),arr(l),arr(c));
}
py::tuple zones(D o,D h,D l,D c,I t,int retrace) {
    if(o.size()!=h.size()||o.size()!=l.size()||o.size()!=c.size()||o.size()!=t.size())
        throw std::runtime_error("OHLC shape mismatch");
    auto z=detect_zones(o.data(),h.data(),l.data(),c.data(),t.data(),t.size(),retrace);
    return py::make_tuple(arr(z.top),arr(z.bottom),arr(z.mid),arr(z.is_buy),arr(z.created_at_ns),arr(z.start_idx));
}
py::tuple trades(D top,D bottom,D mid,py::array_t<uint8_t> buy,I start,I created,I t,D lo,D hi,
                 double age,double n,double rr,int touches) {
    SimResult s;
    {py::gil_scoped_release unlock;
    s=simulate_generic(top.data(),bottom.data(),mid.data(),buy.data(),start.data(),created.data(),top.size(),
        t.data(),lo.data(),hi.data(),t.size(),int64_t(age*3600e9),n,n*rr,false,.5,1.,touches,129600);}
    return py::make_tuple(arr(s.entry_idx),arr(s.exit_idx),arr(s.mid),arr(s.stop_risk),arr(s.is_win),arr(s.touch_num));
}
// Conservative bid/ask execution audit. Resting entry is a limit at zone mid;
// buy entry checks ask, buy exits check bid (sell is symmetric). No candle-wide
// invalidation before an entry which could already have filled. Stops on entry
// candles count; ambiguous entry-candle targets are deferred. Remaining positions
// are liquidated at the last observable close instead of silently omitted.
py::tuple real_trades(D top,D bottom,D mid,py::array_t<uint8_t> buy,I start,I created,I t,
                      D op,D lo,D hi,D cl,double age,double n,double rr,int touches,double spread) {
    std::vector<int64_t> ens,exs,tn;std::vector<double> prices,risks,returns;
    const double *O=op.data(),*L=lo.data(),*H=hi.data(),*C=cl.data();
    {py::gil_scoped_release unlock;
    for(int64_t z=0;z<top.size();z++) {
        int sign=buy.data()[z]?1:-1;double m=mid.data()[z],risk=n*(top.data()[z]-bottom.data()[z])/2;
        double sl=m-sign*risk,tp=m+sign*risk*rr,inv=sign==1?bottom.data()[z]:top.data()[z];
        int64_t j=start.data()[z],end=std::min(j+129600,int64_t(t.size()));
        if(j>=end)continue;
        bool dead=false,active=false;int count=0;int64_t en=-1;
        for(;j<end;j++) {
            double bidlo=L[j]-spread/2,askhi=H[j]+spread/2;
            bool breached=sign==1?L[j]<=inv:H[j]>=inv;
            bool newly=false,filled_open=false;
            if(!active) {
                if(dead||count>=touches)break;
                bool reached=sign==1?L[j]+spread/2<=m:H[j]-spread/2>=m;
                if(reached) {
                    if(count==0&&t.data()[j]-created.data()[z]<int64_t(age*3600e9))break;
                    active=true;newly=true;en=j;++count;
                    filled_open=sign==1?O[j]+spread/2<=m:O[j]-spread/2>=m;
                }
            }
            dead=dead||breached;
            if(active) {
                bool stop=sign==1?bidlo<=sl:askhi>=sl;
                bool target=sign==1?H[j]-spread/2>=tp:L[j]+spread/2<=tp;
                double r=0;bool close=false;
                if(stop) {
                    double worst_open=O[j]-sign*spread/2;
                    // For a non-marketable newly filled limit the opening gap happened
                    // before entry; use stop level. Otherwise charge adverse stop gaps.
                    r=(!newly||filled_open)?std::min(-1.,sign*(worst_open-m)/risk):-1.;close=true;
                } else if(target&&(!newly||filled_open||(sign==1?C[j]-spread/2>=tp:C[j]+spread/2<=tp))) {r=rr;close=true;}
                else if(j==end-1) {r=sign*(C[j]-sign*spread/2-m)/risk;close=true;}
                if(close) {
                    ens.push_back(en);exs.push_back(j);prices.push_back(m);risks.push_back(risk);
                    returns.push_back(r);tn.push_back(count);active=false;
                    if(r<=0||dead||count>=touches)break;
                }
            }
        }
    }}
    return py::make_tuple(arr(ens),arr(exs),arr(prices),arr(risks),arr(returns),arr(tn));
}
// Mechanism-changing experiments: completed-candle reclaim, failed-zone reversal,
// and continuation breakout. Enter at the NEXT open; no entry-candle information
// is used to choose entry. Volatility and structural risk are known before entry.
py::tuple structural_trades(D top,D bottom,D mid,py::array_t<uint8_t> buy,I start,I created,I t,
                            D op,D lo,D hi,D cl,D atr,int mode,double rr,double spread,int max_hold) {
    std::vector<int64_t> ens,exs,tn;std::vector<double> prices,risks,returns;
    const double *O=op.data(),*L=lo.data(),*H=hi.data(),*C=cl.data();
    {py::gil_scoped_release unlock;
    for(int64_t z=0;z<top.size();z++) {
        int original=buy.data()[z]?1:-1,sign=mode==1?-original:original;
        double m=mid.data()[z],far=original==1?bottom.data()[z]:top.data()[z];
        double near=original==1?top.data()[z]:bottom.data()[z];
        int64_t first=std::max(int64_t(2),start.data()[z]);
        int64_t limit=std::min(first+4320,int64_t(t.size())-1);
        // Observe signal candle j; order enters at j+1. A zone gets one opportunity.
        for(int64_t j=first;j<limit;j++) {
            bool beyond=original==1?C[j]<far:C[j]>far;
            bool trigger=false;
            if(mode==0) { // Reclaim with directional body, not just any wick.
                trigger=original==1?(L[j]<=m&&C[j]>m&&C[j]>O[j]):(H[j]>=m&&C[j]<m&&C[j]<O[j]);
                if(beyond)break;
            } else if(mode==1) { // A confirmed close through invalidation reverses direction.
                trigger=beyond&&(original==1?C[j-1]>=far:C[j-1]<=far);
            } else { // Completed break through the favorable zone edge.
                trigger=original==1?(C[j]>near&&C[j-1]<=near):(C[j]<near&&C[j-1]>=near);
                if(beyond)break;
            }
            if(!trigger)continue;
            int64_t en=j+1,last=std::min(en+max_hold,int64_t(t.size())-1);
            double entry=O[en]+sign*spread/2;
            double risk=std::max({(top.data()[z]-bottom.data()[z])/2,2.*atr.data()[j],3.*spread});
            double sl=entry-sign*risk,tp=entry+sign*risk*rr;
            double result=0;int64_t ex=last;
            for(int64_t k=en;k<=last;k++) {
                bool stop=sign==1?L[k]-spread/2<=sl:H[k]+spread/2>=sl;
                bool target=sign==1?H[k]-spread/2>=tp:L[k]+spread/2<=tp;
                if(stop){result=std::min(-1.,sign*(O[k]-sign*spread/2-entry)/risk);ex=k;break;}
                if(target){result=rr;ex=k;break;}
                if(k==last)result=sign*(C[k]-sign*spread/2-entry)/risk;
            }
            ens.push_back(en);exs.push_back(ex);prices.push_back(entry);risks.push_back(risk);returns.push_back(result);tn.push_back(1);
            break;
        }
    }}
    return py::make_tuple(arr(ens),arr(exs),arr(prices),arr(risks),arr(returns),arr(tn));
}
// Forward-only recurrences. Returned row j is known at CLOSE of j, caller uses j-1.
py::array_t<float> features(D o,D h,D l,D c) {
    int64_t n=c.size(); py::array_t<float> result({n,int64_t(14)}); auto f=result.mutable_unchecked<2>();
    const double *O=o.data(),*H=h.data(),*L=l.data(),*C=c.data();
    double a10=0,a60=0,a240=0,e8=0,e32=0,e60=0,e240=0,pressure=0,wick=0,up=0,dn=0,switches=0,ac=0,v=0;
    double abssum=0,prevret=0;
    py::gil_scoped_release unlock;
    for(int64_t j=0;j<n;j++) {
        double pc=j?C[j-1]:C[j], ret=C[j]-pc;
        double tr=std::max(H[j]-L[j],std::max(std::abs(H[j]-pc),std::abs(L[j]-pc)));
        if(j==0) {a10=a60=a240=std::max(tr,1e-12);e8=e32=e60=e240=C[j];}
        auto ew=[](double old,double value,double span){return old+2/(span+1)*(value-old);};
        a10=ew(a10,tr,10);a60=ew(a60,tr,60);a240=ew(a240,tr,240);
        e8=ew(e8,C[j],8);e32=ew(e32,C[j],32);e60=ew(e60,C[j],60);e240=ew(e240,C[j],240);
        pressure=ew(pressure,(C[j]-O[j])/std::max(tr,1e-12),10);
        wick=ew(wick,(2*C[j]-H[j]-L[j])/std::max(H[j]-L[j],1e-12),10);
        up=ew(up,std::max(ret,0.)*std::max(ret,0.),20);dn=ew(dn,std::min(ret,0.)*std::min(ret,0.),20);
        switches=ew(switches,(ret*prevret<0)?1.:0.,60);
        ac=ew(ac,ret*prevret,60);v=ew(v,ret*ret,60);
        abssum+=std::abs(ret); if(j>=30) abssum-=std::abs(C[j-30]-(j>30?C[j-31]:C[0]));
        f(j,0)=pressure;
        f(j,1)=std::abs(C[j]-C[std::max(int64_t(0),j-30)])/std::max(abssum,1e-12);
        f(j,2)=a10/std::max(a240,1e-12);
        f(j,3)=(C[j]-e60)/std::max(a60,1e-12);
        f(j,4)=wick;
        f(j,5)=(up-dn)/std::max(up+dn,1e-12);
        f(j,6)=(e8-e32)/std::max(a60,1e-12);
        f(j,7)=a60/C[j];
        f(j,8)=switches;
        f(j,9)=tr/std::max(a60,1e-12);
        f(j,10)=(C[j]-e240)/std::max(a240,1e-12);
        f(j,11)=wick*a10/std::max(a240,1e-12);
        f(j,12)=(C[j]-C[std::max(int64_t(0),j-60)])/std::max(a60*std::sqrt(60.),1e-12);
        f(j,13)=ac/std::max(v,1e-12);
        prevret=ret;
    }
    return result;
}
// Each query is an entry timestamp. Only foreign bars with open+60 <= query qualify.
py::array_t<double> asof(I query,I times,D close) {
    py::array_t<double> out({int64_t(query.size()),int64_t(3)});auto a=out.mutable_unchecked<2>();
    py::gil_scoped_release unlock;
    for(int64_t i=0;i<query.size();i++) {
        int64_t j=std::upper_bound(times.data(),times.data()+times.size(),query.data()[i]-60000000000LL)-times.data()-1;
        if(j<60||query.data()[i]-times.data()[j]>300000000000LL) {a(i,0)=a(i,1)=a(i,2)=std::numeric_limits<double>::quiet_NaN();continue;}
        a(i,0)=close.data()[j]; a(i,1)=std::log(close.data()[j]/close.data()[j-15]);
        a(i,2)=std::log(close.data()[j]/close.data()[j-60]);
    }
    return out;
}
// Histogram gradient-boosted regression stumps. Feature cutpoints and all leaves
// come exclusively from the supplied training rows; predict never observes labels.
py::tuple learn(D data,D queries,int rounds) {
    auto a=data.unchecked<2>();auto q=queries.unchecked<2>();
    int64_t n=a.shape(0),nf=a.shape(1)-7;int bins=16;
    if(n<1000)throw std::runtime_error("Insufficient closed training outcomes");
    std::vector<std::vector<double>> thresholds(nf);
    std::vector<uint8_t> codes(n*nf);
    std::vector<double> pred(n),output(q.shape(0));
    std::vector<double> model;double mean=0;
    {py::gil_scoped_release unlock;
    for(int64_t i=0;i<n;i++)mean+=std::clamp(a(i,6),-3.,3.);mean/=n;
    std::fill(pred.begin(),pred.end(),mean);std::fill(output.begin(),output.end(),mean);
    for(int64_t f=0;f<nf;f++) {
        std::vector<double> vals;vals.reserve(n);
        for(int64_t i=0;i<n;i++)if(std::isfinite(a(i,f+7)))vals.push_back(a(i,f+7));
        std::sort(vals.begin(),vals.end());
        for(int b=1;b<bins;b++)thresholds[f].push_back(vals.empty()?0:vals[std::min(vals.size()-1,vals.size()*b/bins)]);
        for(int64_t i=0;i<n;i++) {double x=std::isfinite(a(i,f+7))?a(i,f+7):0;
            codes[i*nf+f]=std::upper_bound(thresholds[f].begin(),thresholds[f].end(),x)-thresholds[f].begin();}
    }
    for(int r=0;r<rounds;r++) {
        std::vector<double> sums(nf*bins,0);std::vector<int> counts(nf*bins,0);
        double total=0;
        for(int64_t i=0;i<n;i++) {double residual=std::clamp(a(i,6),-3.,3.)-pred[i];total+=residual;
            for(int64_t f=0;f<nf;f++) {int k=f*bins+codes[i*nf+f];sums[k]+=residual;counts[k]++;}}
        double best=-1,lv=0,rv=0,th=0;int feature=-1;
        for(int64_t f=0;f<nf;f++) {double left=0;int count=0;
            for(int b=0;b<bins-1;b++) {left+=sums[f*bins+b];count+=counts[f*bins+b];
                if(count<std::max(int64_t(256),n/100)||n-count<std::max(int64_t(256),n/100))continue;
                double right=total-left;double gain=left*left/count+right*right/(n-count)-total*total/n;
                if(gain>best){best=gain;feature=f;th=thresholds[f][b];lv=.10*left/count;rv=.10*right/(n-count);}}
        }
        if(feature<0)break;
        model.insert(model.end(),{double(feature+7),th,lv,rv});
        for(int64_t i=0;i<n;i++){double x=std::isfinite(a(i,feature+7))?a(i,feature+7):0;pred[i]+=x<th?lv:rv;}
        for(int64_t i=0;i<q.shape(0);i++){double x=std::isfinite(q(i,feature+7))?q(i,feature+7):0;output[i]+=x<th?lv:rv;}
    }}
    return py::make_tuple(mean,arr(model),arr(output));
}
py::array_t<double> rank_features(D training,D queries) {
    auto a=training.unchecked<2>();auto q=queries.unchecked<2>();
    py::array_t<double> result({q.shape(0),q.shape(1)});auto out=result.mutable_unchecked<2>();
    py::gil_scoped_release unlock;
    for(int64_t i=0;i<q.shape(0);i++)for(int64_t f=0;f<q.shape(1);f++)out(i,f)=q(i,f);
    for(int64_t f=7;f<q.shape(1);f++) {
        std::vector<double> vals;vals.reserve(a.shape(0));
        for(int64_t i=0;i<a.shape(0);i++)if(std::isfinite(a(i,f)))vals.push_back(a(i,f));
        std::sort(vals.begin(),vals.end());
        for(int64_t i=0;i<q.shape(0);i++) {
            if(!std::isfinite(q(i,f))||vals.empty()){out(i,f)=std::numeric_limits<double>::quiet_NaN();continue;}
            auto l=std::lower_bound(vals.begin(),vals.end(),q(i,f))-vals.begin();
            auto h=std::upper_bound(vals.begin(),vals.end(),q(i,f))-vals.begin();
            out(i,f)=.5*(l+h)/vals.size();
        }
    }
    return result;
}
// Metadata col28 is timeframe. Score simultaneous known signals as a group so
// sorting within one timestamp never changes cross-scale consensus.
py::array_t<double> consensus(D data) {
    auto a=data.unchecked<2>();py::array_t<double> result({a.shape(0),py::ssize_t(32)});auto out=result.mutable_unchecked<2>();
    double last[4][2][3];for(auto& instrument:last)for(auto& side:instrument)for(auto& v:side)v=-1e30;
    py::gil_scoped_release unlock;
    for(int64_t i=0;i<a.shape(0);) {
        int64_t end=i+1;while(end<a.shape(0)&&a(end,0)==a(i,0))end++;
        for(int64_t j=i;j<end;j++){int tf=a(j,28)==5?0:(a(j,28)==15?1:2);last[int(a(j,2))][int(a(j,3))][tf]=a(j,0);}
        for(int64_t j=i;j<end;j++) {
            for(int f=0;f<29;f++)out(j,f)=a(j,f);
            int inst=int(a(j,2)),side=int(a(j,3));double same=0,opposite=0;
            for(int tf=0;tf<3;tf++) {double age=a(j,0)-last[inst][side][tf];if(age<=3600)same+=std::exp(-age/3600);
                age=a(j,0)-last[inst][1-side][tf];if(age<=3600)opposite+=std::exp(-age/3600);}
            out(j,29)=same/3;out(j,30)=opposite/3;out(j,31)=(same-opposite)*a(j,9)/3;
        }
        i=end;
    }
    return result;
}
py::array_t<double> confluence_features(D op,D hi,D lo,D cl) {
    int64_t n=cl.size();py::array_t<double> output({n,int64_t(3)});auto out=output.mutable_unchecked<2>();
    std::vector<double> atr(n);double trsum=0,psum=0,p2sum=0,e15=0,e60=0,e240=0;
    std::array<double,14> ring{};
    {py::gil_scoped_release unlock;
    for(int64_t j=0;j<n;j++) {
        double c=cl.data()[j],p=j?cl.data()[j-1]:c;
        double tr=std::max(hi.data()[j]-lo.data()[j],std::max(std::abs(hi.data()[j]-p),std::abs(lo.data()[j]-p)));
        trsum+=tr-ring[j%14];ring[j%14]=tr;atr[j]=trsum/std::min(int64_t(14),j+1);
        psum+=c;p2sum+=c*c;if(j>=20){double old=cl.data()[j-20];psum-=old;p2sum-=old*old;}
        double count=std::min(int64_t(20),j+1),mean=psum/count;
        double stddev=std::sqrt(std::max(0.,p2sum/count-mean*mean));
        out(j,0)=stddev>1e-12?.5+(c-mean)/(4*stddev):.5;
        if(!j)e15=e60=e240=c;
        e15+=2./16*(c-e15);e60+=2./61*(c-e60);e240+=2./241*(c-e240);
        out(j,1)=e15>e60&&e60>e240?1:(e15<e60&&e60<e240?-1:0);
    }
    // Coordinate compression is only an exact order index. Future values change
    // no rolling counts/ranks, verified by prefix invariance tests.
    std::vector<double> values=atr;std::sort(values.begin(),values.end());values.erase(std::unique(values.begin(),values.end()),values.end());
    std::vector<int> tree(values.size()+1,0),ids(n);
    auto add=[&](int k,int delta){for(k++;k<int(tree.size());k+=k&-k)tree[k]+=delta;};
    auto sum=[&](int k){int total=0;for(;k>0;k-=k&-k)total+=tree[k];return total;};
    constexpr int window=20160;
    for(int64_t j=0;j<n;j++) {
        int id=std::lower_bound(values.begin(),values.end(),atr[j])-values.begin();ids[j]=id;add(id,1);
        if(j>=window)add(ids[j-window],-1);
        out(j,2)=double(sum(id+1))/std::min(int64_t(window),j+1);
    }}
    return output;
}
// Trade columns: entry_s,exit_s,instrument,buy,entry,risk,net_r, feature columns...
// Tests: feature column, sign (+1 >= threshold; -1 <= threshold), threshold.
py::dict account(D data,D tests,double risk,double begin,double end,double cap=1e100,bool strict_units=false) {
    auto a=data.unchecked<2>();auto q=tests.unchecked<2>();
    double eq[2]={100,100},pnl[2]={0,0},xt[2]={-1,-1};int ix[2]={-1,-1};
    double peak=200,dd=0,win_sum=0,loss_sum=0,hit=-1;int wins=0,n=0,filtered=0,margin_reject=0;
    std::vector<int64_t> accepted;std::vector<double> trace_t,trace_eq;
    auto close=[&](int s){eq[s]+=pnl[s];double total=eq[0]+eq[1];peak=std::max(peak,total);
        dd=std::min(dd,(total-peak)/peak);if(hit<0&&total>=1e6)hit=(xt[s]-begin)/86400.;
        trace_t.push_back(xt[s]);trace_eq.push_back(total);xt[s]=-1;ix[s]=-1;};
    {py::gil_scoped_release unlock;
    for(int64_t i=0;i<a.shape(0);i++) {
        double et=a(i,0);if(et<begin||et>=end)continue;
        // Purge outcomes outside the evaluated interval; unresolved portfolio trades
        // need separate full bar replay before final acceptance.
        if(a(i,1)>=end)continue;
        for(int k=0;k<2;k++) {int s=(xt[0]>=0&&(xt[1]<0||xt[0]<=xt[1]))?0:1;
            if(xt[s]>=0&&xt[s]<=et)close(s);}
        bool keep=true;for(int64_t k=0;k<q.shape(0);k++) {
            double value=a(i,int(q(k,0)));if(!std::isfinite(value)||q(k,1)*(value-q(k,2))<0){keep=false;break;}}
        if(!keep)continue;filtered++;
        int s=int(a(i,3));double floor_balance=strict_units?0.:1.;
        if(xt[s]>=0||eq[0]<=floor_balance||eq[1]<=floor_balance)continue;
        double base=std::min({eq[0],eq[1],cap});double price=a(i,4),stop=a(i,5);
        int inst=int(a(i,2));double perunit=inst==3?stop/price:(inst==2?stop*1.36:stop);
        double notional=inst==3?1.:(inst==2?price*1.36:price);
        double budget_units=std::min(base*risk/perunit,base*50/notional);
        double units=strict_units?std::floor(budget_units):std::max(1.,budget_units);
        if(units<1){margin_reject++;continue;}
        double margin=units*notional/50;if(margin>std::min(eq[s],cap)){margin_reject++;continue;}
        pnl[s]=units*perunit*a(i,6);xt[s]=a(i,1);ix[s]=i;accepted.push_back(i);n++;
        if(a(i,6)>0){wins++;win_sum+=a(i,6);}else loss_sum-=a(i,6);
    }
    for(int k=0;k<2;k++){int s=(xt[0]>=0&&(xt[1]<0||xt[0]<=xt[1]))?0:1;if(xt[s]>=0)close(s);}}
    double years=(end-begin)/(86400*365.25);py::dict d;
    d["n"]=n;d["wr"]=n?100.*wins/n:0;d["pf"]=loss_sum>0?win_sum/loss_sum:0;
    d["dd"]=100*dd;d["tpy"]=n/years;d["log_growth"]=std::log(std::max((eq[0]+eq[1])/200.,1e-100))/years;
    d["equity"]=eq[0]+eq[1];d["buy_equity"]=eq[1];d["sell_equity"]=eq[0];d["days_to_1m"]=hit;
    d["filtered"]=filtered;d["margin_reject"]=margin_reject;d["accepted"]=arr(accepted);
    d["trace_time"]=arr(trace_t);d["trace_equity"]=arr(trace_eq);return d;
}
#include "nav_account.hpp"
#include "path_trades.hpp"
#include "resolved_features.hpp"
#include "zone_field.hpp"
#include "sweep_samples.hpp"
#include "tick_features.hpp"
#include "quote_context.hpp"
#include "basis_features.hpp"
#include "forecast.hpp"
#include "tree_forecast.hpp"
PYBIND11_MODULE(causal_native,m){m.def("zones",&zones);m.def("trades",&trades);m.def("features",&features);
    m.def("nav_account",&nav_account);
    m.def("path_trades",&path_trades,py::arg("top"),py::arg("bottom"),py::arg("mid"),
          py::arg("buy"),py::arg("start"),py::arg("created"),py::arg("t"),py::arg("op"),
          py::arg("lo"),py::arg("hi"),py::arg("cl"),py::arg("age"),py::arg("n"),
          py::arg("rr"),py::arg("touches"),py::arg("spread"),py::arg("path"),py::arg("require_bounce")=false);
    m.def("resolved_features",&resolved_features);
    m.def("zone_field",&zone_field);
    m.def("sweep_samples",&sweep_samples);
    m.def("tick_features",&tick_features);
    m.def("quote_context",&quote_context);
    m.def("basis_features",&basis_features);
    m.def("forecast_samples",&forecast_samples);m.def("forecast_execute",&forecast_execute);m.def("learn_trees",&learn_trees);
    m.def("aggregate",&aggregate);
    m.def("real_trades",&real_trades);
    m.def("structural_trades",&structural_trades);
    m.def("learn",&learn);
    m.def("rank_features",&rank_features);m.def("consensus",&consensus);
    m.def("confluence_features",&confluence_features);
    m.def("asof",&asof);m.def("account",&account,py::arg("data"),py::arg("tests"),py::arg("risk"),
        py::arg("begin"),py::arg("end"),py::arg("cap")=1e100,py::arg("strict_units")=false);}
