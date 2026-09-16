#include "basis.h"
#include "oblate/occultation.h"
using namespace starry::utils;
int main(int argc,char**argv) {
  int d=std::stoi(argv[1]);double b=std::stod(argv[2]),r=std::stod(argv[3]),f=std::stod(argv[4]),theta=std::stod(argv[5]);
  using AD=ADScalar<double,4>;auto ad=[](double x){AD a(x);a.derivatives().setZero();return a;};
  auto variable=[&](double x,int i){AD a=ad(x);a.derivatives()(i)=1.;return a;};
  starry::oblate::occultation::Occultation<double,4> o(d);o.compute(variable(b,0),variable(r,1),variable(f,2),variable(theta,3));
  Eigen::SparseMatrix<double>a1,a2,a;starry::basis::computeA1(d,a1,2./std::sqrt(M_PI));starry::basis::computeA(d,a1,a2,a);
  RowVector<AD> row=o.sT*a;std::cout<<std::setprecision(17);for(int i=0;i<row.size();i++)std::cout<<row(i).value()<<" ";std::cout<<"\n";
  if(argc>6) {for(int j=0;j<4;j++){for(int i=0;i<row.size();i++)std::cout<<row(i).derivatives()(j)<<" ";std::cout<<"\n";}}
}
