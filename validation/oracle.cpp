// Test-only harness. Links the unmodified pinned upstream headers, not Rust.
#include "basis.h"
#include "solver.h"
#include "limbdark.h"
#include "wigner.h"
#include <string>
using namespace starry::utils;
int main(int argc,char**argv) {
  std::cout << std::setprecision(17);
  std::string mode=argc>1?argv[1]:"";
  int degree=argc>2?std::stoi(argv[2]):0;
  int n=(degree+1)*(degree+1);
  if(mode=="basis") {
    Eigen::SparseMatrix<double> a1,a2,a2i;
    starry::basis::computeA1(degree,a1,2./std::sqrt(M_PI));
    starry::basis::computeA2(degree,a2,a2i);
    for(int i=0;i<n;i++)for(int j=0;j<n;j++)std::cout<<a1.coeff(i,j)<<" ";
    for(int i=0;i<n;i++)for(int j=0;j<n;j++)std::cout<<a2i.coeff(i,j)<<" ";
    RowVector<double> rt;
    starry::basis::computerT(degree,rt);
    for(int i=0;i<n;i++)std::cout<<rt(i)<<" ";
  } else if(mode=="flux" || mode=="gradient") {
    double b=std::stod(argv[3]),r=std::stod(argv[4]);
    Eigen::SparseMatrix<double> a1,a2,a;
    starry::basis::computeA1(degree,a1,2./std::sqrt(M_PI));
    starry::basis::computeA(degree,a1,a2,a);
    starry::solver::Greens<double> solver(degree);
    RowVector<double> row(n);
    if(mode=="gradient") {
      solver.compute<true>(b,r);
      RowVector<double> db=solver.dsTdb*a,dr=solver.dsTdr*a;
      for(int i=0;i<n;i++)std::cout<<db(i)<<" ";
      for(int i=0;i<n;i++)std::cout<<dr(i)<<" ";
      std::cout<<"\n";return 0;
    } else if(r==0. || b>=1.+r) {
      RowVector<double> rt; starry::basis::computerT(degree,rt); row=rt*a1;
    } else if(r>=1.+b) {row.setZero();}
    else {solver.compute(b,r);row=solver.sT*a;}
    for(int i=0;i<n;i++)std::cout<<row(i)<<" ";
  } else if(mode=="limb") {
    int total=degree+2, nt=(total+1)*(total+1);
    double b=std::stod(argv[3]),r=std::stod(argv[4]);
    Eigen::SparseMatrix<double> a1,a2,a,u1;
    starry::basis::computeA1(total,a1,2./std::sqrt(M_PI));
    starry::basis::computeA(total,a1,a2,a);
    starry::basis::computeU(total,a1,a,u1,2./std::sqrt(M_PI));
    Vector<double> u=Vector<double>::Zero(total+1);u(0)=-1.;u(1)=.4;u(2)=.2;
    Vector<double> pu=u1*u;
    RowVector<double> rt;starry::basis::computerT(total,rt);
    pu*=M_PI/rt.dot(pu);
    RowVector<double> moments(nt);
    if(r==0. || b>=1.+r) moments=rt;
    else if(r>=1.+b) moments.setZero();
    else {starry::solver::Greens<double> solver(total);solver.compute(b,r);moments=solver.sT*a2;}
    using Triplet=Eigen::Triplet<double>;
    std::vector<Triplet> limb;
    for(int l=0;l<=2;l++)for(int m=-l;m<=l;m++)if(pu(l*l+l+m)!=0.)limb.push_back(Triplet(l,m,pu(l*l+l+m)));
    for(int col=0;col<n;col++) {
      std::vector<Triplet> p,product;
      for(int l=0;l<=degree;l++)for(int m=-l;m<=l;m++) {
        double v=a1.coeff(l*l+l+m,col);if(v!=0.)p.push_back(Triplet(l,m,v));
      }
      starry::basis::computeSparsePolynomialProduct(p,limb,product);
      double value=0.;for(auto t:product)value+=moments(t.row()*t.row()+t.row()+t.col())*t.value();
      std::cout<<value<<" ";
    }
  } else if(mode=="rotation") {
    double x=std::stod(argv[3]),y=std::stod(argv[4]),z=std::stod(argv[5]),t=std::stod(argv[6]);
    starry::wigner::Wigner<double> w(degree,0,0);
    Matrix<double> id=Matrix<double>::Identity(n,n);
    // maps.py Map.rotate uses the RHS operator at -theta.
    w.dotR(id,x,y,z,-t);
    for(int i=0;i<n;i++)for(int j=0;j<n;j++)std::cout<<w.dotR_result(i,j)<<" ";
  } else if(mode=="cel") {
    double k=std::stod(argv[3]),p=std::stod(argv[4]),a=std::stod(argv[5]),b=std::stod(argv[6]);
    std::cout<<starry::ellip::CEL(k,p,a,b);
  } else return 2;
  std::cout<<"\n";
}
