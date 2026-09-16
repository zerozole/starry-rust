// Unmodified upstream reflected phase and occultation kernels; test only.
#include "reflected/occultation.h"
using namespace starry::utils;
int main(int argc,char**argv) {
  int d=std::stoi(argv[1]);double b=std::stod(argv[2]),theta=std::stod(argv[3]),bo=std::stod(argv[4]),ro=std::stod(argv[5]),rough=std::stod(argv[6]);
  using AD=ADScalar<double,5>;
  auto ad=[](double x){AD a(x);a.derivatives().setZero();return a;};
  auto variable=[&](double x,int i){AD a=ad(x);a.derivatives()(i)=1.;return a;};
  starry::basis::Basis<double> basis(d,0,0);
  RowVector<AD> row;
  if(ro==0.) {
    starry::reflected::phasecurve::PhaseCurve<AD> p(d,basis);p.compute(variable(b,0),variable(rough,4));
    row=p.rT*basis.A1;
  } else {
    starry::reflected::occultation::Occultation<AD> p(d,basis);p.compute(variable(b,0),variable(theta,1),variable(bo,2),variable(ro,3),variable(rough,4));
    row=p.sT*basis.A;
  }
  std::cout<<std::setprecision(17);for(int i=0;i<row.size();i++)std::cout<<row(i).value()<<" ";std::cout<<"\n";
  if(argc>7) { for(int j=0;j<5;j++) {for(int i=0;i<row.size();i++)std::cout<<row(i).derivatives()(j)<<" ";std::cout<<"\n";} }
}
