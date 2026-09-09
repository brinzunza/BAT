#pragma once
// Join complete provider minutes strictly preceding entry. Minute end == entry
// is allowed because all underlying ticks are strictly before the minute end.
py::array_t<double> quote_context(D samples,std::vector<D> minutes) {
    auto a=samples.unchecked<2>();py::array_t<double> output({a.shape(0),py::ssize_t(20)});
    auto out=output.mutable_unchecked<2>();py::gil_scoped_release unlock;
    for(int64_t i=0;i<a.shape(0);i++) {
        for(int f=0;f<20;f++)out(i,f)=std::numeric_limits<double>::quiet_NaN();
        int inst=int(a(i,2));double time=a(i,0),sign=a(i,3)?1:-1;
        auto m=minutes[inst].unchecked<2>();int64_t l=0,h=m.shape(0);
        while(l<h){int64_t k=(l+h)/2;if(m(k,0)<=time)l=k+1;else h=k;}
        int64_t last=l-1;if(last<0||m(last,0)!=time)continue;
        int widths[3]={1,5,15};bool valid=true;
        for(int w=0;w<3;w++) {
            int width=widths[w];int64_t first=last-width+1;
            if(first<0){valid=false;break;}
            double n=0,direction=0,imb=0,micro=0,spread=0,variation=0;
            for(int64_t j=first;j<=last;j++) {
                if(m(j,0)!=time-(last-j)*60||m(j,1)<=0){valid=false;break;}
                double count=m(j,1);n+=count;direction+=count*m(j,2);imb+=count*m(j,3);
                micro+=count*m(j,4);spread+=count*m(j,5);variation+=m(j,8)*m(j,8);
            }
            if(!valid)break;
            out(i,w*6)=n/width;out(i,w*6+1)=sign*direction/n;out(i,w*6+2)=sign*imb/n;
            out(i,w*6+3)=sign*micro/n;out(i,w*6+4)=spread/n;out(i,w*6+5)=std::sqrt(variation);
        }
        if(valid){out(i,18)=out(i,0)/std::max(out(i,12),1e-12);out(i,19)=out(i,2)-out(i,14);}
        else for(int f=0;f<20;f++)out(i,f)=std::numeric_limits<double>::quiet_NaN();
    }
    return output;
}
