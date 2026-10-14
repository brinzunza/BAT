#pragma once
// False structural break followed by completed-candle recapture. One opportunity
// per confirmed zone; the stop is based on the already observed sweep extreme.
py::array_t<double> sweep_samples(D top,D bottom,py::array_t<uint8_t> buy,I start,I t,
                                 D op,D lo,D hi,D cl,D atr,double spread,int minutes) {
    std::vector<double> rows;
    py::gil_scoped_release unlock;
    for(int64_t z=0;z<top.size();z++) {
        int sign=buy.data()[z]?1:-1;int64_t first=start.data()[z];
        if(first<=0||first>=t.size()-1)continue;
        double boundary=sign==1?bottom.data()[z]:top.data()[z],extreme=boundary;
        int64_t breached=-1;
        for(int64_t j=first;j<t.size()-1;j++) {
            if(t.data()[j]-t.data()[first]>3LL*86400*1000000000LL)break;
            if(breached<0&&(sign==1?lo.data()[j]<boundary:hi.data()[j]>boundary))breached=j;
            if(breached<0)continue;
            if(t.data()[j]-t.data()[breached]>int64_t(3*minutes)*60000000000LL)break;
            extreme=sign==1?std::min(extreme,lo.data()[j]):std::max(extreme,hi.data()[j]);
            bool reclaimed=sign==1?(cl.data()[j]>boundary&&cl.data()[j]>op.data()[j]&&cl.data()[j]<top.data()[z]):
                (cl.data()[j]<boundary&&cl.data()[j]<op.data()[j]&&cl.data()[j]>bottom.data()[z]);
            if(!reclaimed)continue;
            double entry=op.data()[j+1]+sign*spread/2;
            double risk=std::max(3*spread,sign*(entry-extreme)+spread/2);
            double scale=std::max(atr.data()[j],1e-10),range=std::max(hi.data()[j]-lo.data()[j],1e-10);
            double recovery=sign==1?(cl.data()[j]-lo.data()[j])/range:(hi.data()[j]-cl.data()[j])/range;
            double time=t.data()[j+1]/1e9;
            rows.insert(rows.end(),{time,time,0.,double(sign==1),entry,risk,0.,
                std::abs(boundary-extreme)/scale,recovery,double(j-breached),risk/scale,(top.data()[z]-bottom.data()[z])/scale});
            break;
        }
    }
    py::gil_scoped_acquire lock;
    py::array_t<double> out({py::ssize_t(rows.size()/12),py::ssize_t(12)});
    std::copy(rows.begin(),rows.end(),out.mutable_data());return out;
}
