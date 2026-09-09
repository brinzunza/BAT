#pragma once
// Spatial field of confirmed, still-live zones. Updates on completed candles;
// an invalidating wick precedes any zones newly confirmed by that candle.
py::array_t<double> zone_field(D top,D bottom,py::array_t<uint8_t> buy,I start,I t,
                              D lo,D hi,D cl,D atr,I queries,int minutes) {
    py::array_t<double> result({queries.size(),py::ssize_t(10)});auto out=result.mutable_unchecked<2>();
    std::vector<int64_t> order(top.size()),active;std::iota(order.begin(),order.end(),0);
    const int64_t step=int64_t(minutes)*60000000000LL,ttl=7LL*86400*1000000000LL;
    int64_t bar=0,p=0;
    py::gil_scoped_release unlock;
    std::stable_sort(order.begin(),order.end(),[&](int64_t a,int64_t b){return start.data()[a]<start.data()[b];});
    for(int64_t q=0;q<queries.size();q++) {
        int64_t now=queries.data()[q];
        while(bar<t.size()&&t.data()[bar]+step<=now) {
            active.erase(std::remove_if(active.begin(),active.end(),[&](int64_t z){
                int64_t born=start.data()[z];
                return t.data()[bar]-(born>0?t.data()[born-1]+step:t.data()[0])>=ttl||
                    (buy.data()[z]?lo.data()[bar]<=bottom.data()[z]:hi.data()[bar]>=top.data()[z]);
            }),active.end());
            while(p<int64_t(order.size())&&start.data()[order[p]]<=bar+1) {
                int64_t z=order[p++],born=start.data()[z];
                if(born>=t.size())continue;
                if(buy.data()[z]?cl.data()[bar]>bottom.data()[z]:cl.data()[bar]<top.data()[z])active.push_back(z);
            }
            bar++;
        }
        for(int f=0;f<10;f++)out(q,f)=0;
        out(q,4)=out(q,5)=8.;
        if(!bar)continue;
        double c=cl.data()[bar-1],scale=std::max(atr.data()[bar-1],1e-10),sum=0,sum2=0;
        for(int64_t z:active) {
            double mid=(top.data()[z]+bottom.data()[z])/2,d=(mid-c)/scale;
            double w=std::exp(-std::abs(d));out(q,buy.data()[z]?0:1)+=w;
            out(q,d>=0?2:3)+=w;
            out(q,d>=0?4:5)=std::min(out(q,d>=0?4:5),std::abs(d));
            int64_t born=start.data()[z];
            if(now-(born>0?t.data()[born-1]+step:t.data()[0])<=6LL*3600*1000000000LL)out(q,7)+=1;
            sum+=d;sum2+=d*d;out(q,9)+=d*w;
        }
        double n=active.size();out(q,6)=std::log1p(n);out(q,7)=std::log1p(out(q,7));
        out(q,8)=n?std::sqrt(std::max(0.,sum2/n-(sum/n)*(sum/n))):0;
    }
    return result;
}
