#pragma once
// Confirmed-zone forecasts: all predictors known before next-open execution.
py::array_t<double> forecast_samples(D top,D bottom,D mid,py::array_t<uint8_t> buy,I start,I created,
                                   I times,D close,D features,D raw,int horizon,int timeframe,double spread) {
    auto f=features.unchecked<2>();auto r=raw.unchecked<2>();
    std::vector<double> rows;
    {py::gil_scoped_release unlock;
    for(int64_t z=0;z<top.size();z++) {
        int64_t j=start.data()[z];if(j<240||j>=times.size())continue;
        double target=times.data()[j]/1e9;int64_t l=0,h=r.shape(0);
        while(l<h){int64_t k=(l+h)/2;if(r(k,0)<target)l=k+1;else h=k;}
        int64_t en=l,ex=en+horizon-1;if(ex>=r.shape(0))continue;
        double atr=f(j-1,7)*close.data()[j-1],half=(top.data()[z]-bottom.data()[z])/2;
        double risk=std::max({half,2*atr,3*spread});int sign=buy.data()[z]?1:-1;
        double t=r(en,0),price=r(en,1),label=sign*(r(ex,4)-price)/risk;
        rows.insert(rows.end(),{t,r(ex,0)+60,0.,double(buy.data()[z]),price,risk,label});
        for(int k=0;k<14;k++){bool oriented=k==0||k==3||k==4||k==5||k==6||k==10||k==11||k==12;rows.push_back(f(j-1,k)*(oriented?sign:1));}
        double hour=std::fmod(t,86400.)/86400.,day=std::fmod(std::floor(t/86400.)+3,7.);
        rows.insert(rows.end(),{half/std::max(atr,1e-12),sign*(close.data()[j-1]-mid.data()[z])/std::max(atr,1e-12),
             std::log1p((t-created.data()[z]/1e9)/60),std::sin(6.28318530718*hour),std::cos(6.28318530718*hour),
             std::sin(6.28318530718*day/7),std::cos(6.28318530718*day/7),double(timeframe),spread/risk});
    }}
    py::array_t<double> out({int64_t(rows.size()/30),int64_t(30)});std::copy(rows.begin(),rows.end(),out.mutable_data());return out;
}
// Execute a predicted direction at its known next-open sample, with a native
// bid/ask bracket. Zero predicted direction skips. Stop wins same-minute ties.
py::array_t<double> forecast_execute(D samples,D directions,std::vector<D> markets,int horizon,double rr) {
    auto a=samples.unchecked<2>();std::vector<double> result;
    {py::gil_scoped_release unlock;
    for(int64_t i=0;i<a.shape(0);i++) {
        if(directions.data()[i]==0)continue;
        int inst=int(a(i,2)),original=a(i,3)?1:-1,sign=directions.data()[i]>0?original:-original;
        auto r=markets[inst].unchecked<2>();int64_t low=0,high=r.shape(0);
        while(low<high){int64_t k=(low+high)/2;if(r(k,0)<a(i,0))low=k+1;else high=k;}
        int64_t en=low,last=std::min(en+horizon-1,int64_t(r.shape(0)-1));if(en>=r.shape(0))continue;
        double spread=inst==3?.016:.00016,entry=r(en,1)+sign*spread/2,risk=a(i,5);
        double sl=entry-sign*risk,tp=entry+sign*risk*rr,net=0;int64_t ex=last;
        for(int64_t j=en;j<=last;j++) {
            bool stop=sign==1?r(j,3)-spread/2<=sl:r(j,2)+spread/2>=sl;
            bool target=sign==1?r(j,2)-spread/2>=tp:r(j,3)+spread/2<=tp;
            if(stop){net=std::min(-1.,sign*(r(j,1)-sign*spread/2-entry)/risk);ex=j;break;}
            if(target&&rr>0){net=rr;ex=j;break;}
            if(j==last)net=sign*(r(j,4)-sign*spread/2-entry)/risk;
        }
        // Preserve original feature columns for forecasting audit; account uses 0:7 only.
        result.insert(result.end(),{a(i,0),r(ex,0)+60,double(inst),double(sign==1),entry,risk,net});
        for(int k=7;k<a.shape(1);k++)result.push_back(a(i,k));
    }}
    py::array_t<double> out({py::ssize_t(result.size()/a.shape(1)),a.shape(1)});std::copy(result.begin(),result.end(),out.mutable_data());return out;
}
