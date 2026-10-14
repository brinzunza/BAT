#pragma once
// Cross-currency graph residual, based only on already-completed E/G/R candles.
// Output: z-score; three raw fair-value distances in pips; three demeaned distances.
py::array_t<double> basis_features(std::vector<D> markets,D query) {
    if(markets.size()<3)throw std::runtime_error("EURUSD, GBPUSD, EURGBP arrays required");
    py::array_t<double> result({int64_t(query.size()),int64_t(7)});auto out=result.mutable_unchecked<2>();
    const double* m[3];int64_t len[3],ptr[3]={0,0,0};
    for(int k=0;k<3;k++){m[k]=markets[k].data();len[k]=markets[k].shape(0);}
    double mean=0,var=0,latest=-1e30,values[7];bool initialized=false;
    std::fill(values,values+7,std::numeric_limits<double>::quiet_NaN());
    {py::gil_scoped_release unlock;
    for(int64_t row=0;row<query.size();row++) {
        double now=query.data()[row];
        while(ptr[0]<len[0]&&m[0][ptr[0]*5]+60<=now) {
            int64_t e=ptr[0]++;double time=m[0][e*5]+60;
            for(int k=1;k<3;k++)while(ptr[k]<len[k]&&m[k][ptr[k]*5]+60<=time)ptr[k]++;
            if(ptr[1]==0||ptr[2]==0)continue;
            if(time-(m[1][(ptr[1]-1)*5]+60)>60||time-(m[2][(ptr[2]-1)*5]+60)>60)continue;
            double E=m[0][e*5+4],G=m[1][(ptr[1]-1)*5+4],R=m[2][(ptr[2]-1)*5+4];
            double residual=std::log(E/(G*R));
            if(!initialized){mean=residual;initialized=true;}
            double innovation=residual-mean;mean+=2./121*innovation;var=(1-2./121)*(var+2./121*innovation*innovation);
            values[0]=innovation/std::sqrt(std::max(var,1e-16));
            values[1]=(G*R-E)/.0001;values[2]=(E/R-G)/.0001;values[3]=(E/G-R)/.0001;
            values[4]=-innovation*E/.0001;values[5]=innovation*G/.0001;values[6]=innovation*R/.0001;
            latest=time;
        }
        for(int f=0;f<7;f++)out(row,f)=now-latest<=120?values[f]:std::numeric_limits<double>::quiet_NaN();
    }}
    return result;
}
