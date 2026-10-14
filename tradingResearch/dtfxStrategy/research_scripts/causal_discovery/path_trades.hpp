#pragma once
// Two explicit, admissible OHLC paths. No use of later candle extrema to veto
// an earlier fill. Cancellation is an event at the structural boundary.
py::tuple path_trades(D top,D bottom,D mid,py::array_t<uint8_t> buy,I start,I created,I t,
                     D op,D lo,D hi,D cl,double age,double n,double rr,int touches,double spread,int path,
                     bool require_bounce=false) {
    std::vector<int64_t> ens,exs,tn;std::vector<double> prices,risks,returns;
    {py::gil_scoped_release unlock;
    for(int64_t z=0;z<top.size();z++) {
        int sign=buy.data()[z]?1:-1;double m=mid.data()[z],risk=n*(top.data()[z]-bottom.data()[z])/2;
        double invalid=sign==1?bottom.data()[z]:top.data()[z];
        double entry_level=m-sign*spread/2,sl_level=m-sign*risk+sign*spread/2,tp_level=m+sign*risk*rr+sign*spread/2;
        int64_t first=start.data()[z],end=std::min(first+129600,int64_t(t.size()));
        if(first>=end)continue;
        if(first>0&&(sign==1?cl.data()[first-1]<=invalid:cl.data()[first-1]>=invalid))continue;
        bool active=false,dead=false,stop_zone=false,resting=false;
        bool in_probe=false,had_bounce=false;
        const double excursion=sign==1?top.data()[z]+(top.data()[z]-bottom.data()[z]):bottom.data()[z]-(top.data()[z]-bottom.data()[z]);
        int count=0;int64_t entry_bar=-1;
        for(int64_t j=first;j<end;j++) {
            bool done_bar=false;
            // Repeat orders become eligible after a completed-candle bounce.
            // This deliberately avoids interpreting unknown intra-minute bot dispatch.
            bool bounce_ready=!require_bounce||count==0||had_bounce;
            bool mature=t.data()[j]-created.data()[z]>=int64_t(age*3600e9)||count>0;
            auto invalid_at=[&](double p){return sign==1?p<=invalid:p>=invalid;};
            auto enter_at=[&](double p){return sign==1?p<=entry_level:p>=entry_level;};
            auto emit=[&](double r){ens.push_back(entry_bar);exs.push_back(j);prices.push_back(m);risks.push_back(risk);
                returns.push_back(r);tn.push_back(count);active=false;done_bar=true;
                in_probe=false;had_bounce=false;if(r<=0)stop_zone=true;};
            auto check_at=[&](double p,bool gap) {
                if(!active)return;
                bool sl=sign==1?p<=sl_level:p>=sl_level;
                bool tp=sign==1?p>=tp_level:p<=tp_level;
                if(sl)emit(gap?std::min(-1.,sign*(p-sign*spread/2-m)/risk):-1.);
                else if(tp)emit(rr);
            };
            double o=op.data()[j];
            if(active)check_at(o,true);
            else if(!dead&&!stop_zone&&count<touches&&mature&&bounce_ready) {
                // An order already resting may fill across a gap before a bot can
                // cancel. A newly eligible order cannot knowingly enter an invalid zone.
                if(enter_at(o)&&(resting||!invalid_at(o))){active=true;entry_bar=j;++count;check_at(o,true);}
            }
            if(invalid_at(o))dead=true;
            double points[4]={o,path==0?hi.data()[j]:lo.data()[j],path==0?lo.data()[j]:hi.data()[j],cl.data()[j]};
            for(int segment=0;segment<3;segment++) {
                double p=points[segment],q=points[segment+1];
                if(!active&&!done_bar&&!dead&&!stop_zone&&count<touches) {
                    if(!mature) {
                        if(sign==1?std::min(p,q)<=m:std::max(p,q)>=m)dead=true;
                    } else if(bounce_ready) {
                        bool crosses=sign==1?(q<=entry_level&&p>entry_level):(q>=entry_level&&p<entry_level);
                        if(crosses) {
                            // Along this directed segment, whichever boundary is
                            // encountered first acts first. No hindsight candle veto.
                            bool cancel_first=sign==1?invalid>=entry_level:invalid<=entry_level;
                            if(cancel_first)dead=true;
                            else {active=true;entry_bar=j;++count;p=entry_level;check_at(p,false);}
                        }
                    }
                }
                if(active) {
                    // A monotone segment can hit only the next barrier in its direction.
                    if(q>p) {
                        if(sign==1&&q>=tp_level)emit(rr);
                        else if(sign==-1&&q>=sl_level)emit(-1.);
                    } else if(q<p) {
                        if(sign==1&&q<=sl_level)emit(-1.);
                        else if(sign==-1&&q<=tp_level)emit(rr);
                    }
                }
                if(invalid_at(q))dead=true;
                if(require_bounce&&!dead&&!active&&!done_bar&&count>0) {
                    if(sign==1?p<=m:p>=m)in_probe=true;
                    if(in_probe&&(sign==1?q>=excursion:q<=excursion))had_bounce=true;
                    if(sign==1?q<=m:q>=m)in_probe=true;
                }
            }
            if(active&&j==end-1)emit(sign*(cl.data()[j]-sign*spread/2-m)/risk);
            if(!active&&(dead||stop_zone||count>=touches))break;
            resting=!active&&!dead&&mature&&!stop_zone&&(!require_bounce||count==0||had_bounce);
        }
    }}
    return py::make_tuple(arr(ens),arr(exs),arr(prices),arr(risks),arr(returns),arr(tn));
}
