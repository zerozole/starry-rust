"""Independent 50-digit scalar harmonic integrals using Python Decimal.

Checks nonzero azimuthal orders at +y occultor positions. Uses unnormalized
associated Legendre polynomials, factorial normalization and Chebyshev angular
integrals, independent of Rust's normalized recurrence and adaptive integrator.
No external high-precision package is required.
"""
from decimal import Decimal as D,localcontext
from functools import lru_cache
import json
import math
from compare import ROOT,ORACLE,RUST,values

PI=D('3.1415926535897932384626433832795028841971693993751058209749445923')

def sin(x):
    term=x;total=x
    for n in range(1,100):
        term *= -x*x/D((2*n)*(2*n+1));new=total+term
        if new==total:return total
        total=new
    raise ArithmeticError('sine failed to converge')

def atan(x):
    if x>D(1):return PI/2-atan(1/x)
    for _ in range(2):x=x/(1+(1+x*x).sqrt())
    term=x;total=x
    for n in range(1,150):
        term *= -x*x;next_total=total+term/D(2*n+1)
        if next_total==total:return 4*total
        total=next_total
    raise ArithmeticError('atan failed to converge')

def legendre(n,x):
    previous=D(0);value=D(1)
    for k in range(1,n+1):
        previous,value=value,(D(2*k-1)*x*value-D(k-1)*previous)/D(k)
    return value,previous

@lru_cache(None)
def gauss(n):
    result=[]
    for i in range(n):
        z=D(str(math.cos(math.pi*(i+.75)/(n+.5))))
        for _ in range(30):
            p,previous=legendre(n,z);derivative=D(n)*(z*p-previous)/(z*z-1)
            next_z=z-p/derivative
            if abs(next_z-z)<D('1e-47'):break
            z=next_z
        z=next_z;p,previous=legendre(n,z);derivative=D(n)*(z*p-previous)/(z*z-1)
        result.append((z,D(2)/((1-z*z)*derivative*derivative)))
    return result

def integral(l,m,b,r,n):
    b=D(str(b));r=D(str(r));order=abs(m)
    lo=abs(b-r);hi=min(D(1),b+r)
    if hi<=lo:return D(0)
    azimuth=([0,1,0,-1] if m<0 else [1,0,-1,0])[order%4]
    norm=(D((2 if order else 1)*(2*l+1))*D(math.factorial(l-order))/D(math.factorial(l+order))).sqrt()/PI
    total=D(0)
    for node,weight in gauss(n):
        t=PI*(node+1)/4
        rho=lo+(hi-lo)*sin(t)**2
        mu=((1-rho)*(1+rho)).sqrt()
        p=D(math.prod(range(1,2*order,2)))*rho**order;previous=D(0)
        for degree in range(order+1,l+1):
            previous,p=p,(D(2*degree-1)*mu*p-D(degree+order-1)*previous)/D(degree-order)
        cosine=(rho*rho+b*b-r*r)/(2*rho*b)
        sine=(1-cosine*cosine).sqrt()
        previous_s=D(0);s=sine
        for _ in range(2,order+1):previous_s,s=s,2*cosine*s-previous_s
        angular=-2*s/D(order)*azimuth if order else 2*(PI-2*atan(((1-cosine)/(1+cosine)).sqrt()))
        total+=weight*PI/4*rho*(hi-lo)*sin(2*t)*norm*p*angular
    if order==0:
        for a,z in ([(D(0),lo)] if b>r else [])+([(hi,D(1))] if hi<1 else []):
            mu_lo=(1-z*z).sqrt();mu_hi=(1-a*a).sqrt()
            for node,weight in gauss(n):
                mu=(mu_hi+mu_lo+(mu_hi-mu_lo)*node)/2
                total+=weight*(mu_hi-mu_lo)*PI*norm*mu*legendre(l,mu)[0]
    return total

def main():
    checks=[(5,-1,.3,.1),(5,2,.7,.4),(15,-1,.01,1.),(20,6,.5,.5),(20,4,.01,1.)]
    results=[]
    with localcontext() as context:
        context.prec=50
        for l,m,b,r in checks:
            coarse=integral(l,m,b,r,64);fine=integral(l,m,b,r,128)
            index=l*l+l+m
            native=values(RUST,'rings',l,b,r)[index]
            upstream=values(ORACLE,'flux',l,b,r)[index]
            record=dict(degree=l,order=m,b=b,r=r,decimal_reference=str(fine),
                quadrature_difference=float(abs(coarse-fine)),rust_difference=abs(native-float(fine)),
                upstream_difference=abs(upstream-float(fine)))
            record['passed']=record['quadrature_difference']<1e-20 and record['rust_difference']<1e-11
            results.append(record)
    report=dict(precision_digits=50,cases=results,passed=all(r['passed'] for r in results))
    (ROOT/'validation'/'decimal_rings_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()
