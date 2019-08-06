#include "mfem.hpp"
/**********************************************
 * Define the source term
 * source(r,z,u) = F(r,z,u)/r
 *				 = nonlinaer_source(r,z,u)
 *				  +linear_source(r,z)
 *
 * Also, define the exact solution/dirichelet 
 * boundary condition:
 *		u_exact (Dirichlet boundary condition)
 *		, q_exact
 * *********************************************/

using namespace mfem;
using namespace std;

/***********************************************/
/***********************************************/
/***********************************************/

// source terms 
// source = nonlienar_source + linear_source


int sol_opt = 0; /* decide which source term to use */

/*************************************
 * nonlinear_source gives the nonlinear part of the source
 *
 * derivative_of_nonlinear_source gives its derivative corresponding
 * to u
 * ***********************************/
double nonlinear_source(double u, const Vector &x)
{
	if( sol_opt == 0){
		return u - u*u -  exp(-u);
	}
	else if (sol_opt == 3){	
		double kr = 1.15*M_PI, kz = 1.15;
		return (kr*kr+kz*kz)/x(0)*u -u*u - exp(-u);
	}
	else if (sol_opt == 4){	
//		double sigma2=0.05, c1 = 0.8, c2 = 0.2;
		double sigma2=0.005, c1 = 0.8, c2 = 0.2;// really really bad convergence
		double r=x(0), res = 0.;

		res = 2.*r*u*
			  (
			   c2 * (1-exp(-u*u/sigma2) ) 
			  +1./sigma2 * (c1+c2*u*u) * exp(-u*u/sigma2)
			  );
		return res;
	}
	else if (sol_opt == 5){
		double r = x(0);
		return r*(1. - 0.5 * (1-u*u)*(1-u*u) );
    }
	else{
		return u - 0.5*u*u;
	}
}


double derivative_of_nonlinear_source(double u,const Vector &x)
{
	if( sol_opt == 0){
		return 1 - 2.*u +  exp(-u);
	}
	else if (sol_opt == 3){	
		double kr = 1.15*M_PI, kz = 1.15;
		return (kr*kr+kz*kz)/x(0) -2*u + exp(-u);
	}
	else if(sol_opt == 4){
		double sigma2=0.05, c1 = 0.8, c2 = 0.2; // fine
//		double sigma2=0.005, c1 = 0.8, c2 = 0.2;// really really bad convergence
		double r=x(0), res = 0.;

		res = 2.*r*(
				c2*(1-exp(-u*u/sigma2)  )
			    +1./sigma2 * (c1+c2*u*u) * exp(-u*u/sigma2) 
			  )
			 +4.*r*c2*u/sigma2*exp(-u*u/sigma2)
			 -4.*r*u/sigma2/sigma2* (c1 + c2*u*u) * exp(-u*u/sigma2)
			 +4.*r*u/sigma2* c2*u*exp(-u*u/sigma2)
			 ;
		
//		res = 
//			 2.*r*( c2 * ( 1 - exp(- u*u/sigma2) )
//			  +1./sigma2 * (c1+c2*u*u) * exp(-u*u/sigma2)
//			 )
//			 +2.*r*u*
//			 ( c2 * 2.*u/sigma2 * exp(- u*u/sigma2) 
//			  -2.*u/sigma2/sigma2 * (c1+c2*u*u) * exp(-u*u/sigma2)
//			  +2./sigma2 * c2*u * exp(-u*u/sigma2)
//			 );
		return res;
	}
	else if(sol_opt == 5){
		double r=x(0);
    	return r*(1-u*u)*(1-u*u)* 2 * u;
    }
	else{
		return 1. - u;
	}
}
/****************************************************/
double nonlinear_source(double u)
{
	if( sol_opt == 0){
		return u - u*u -  exp(-u);
	}
	else if (sol_opt == 3){	
		return -u*u - exp(-u);
	}
	else{
		return u - 0.5*u*u;
	}
}


double derivative_of_nonlinear_source(double u)
{
	if( sol_opt == 0){
		return 1 - 2.*u +  exp(-u);
	}
	else if (sol_opt == 3){	
		return -2*u + exp(-u);
	}
	else{
		return 1. - u;
	}
}

/************************************************
 * exact solution/boundary condition u_exact, 
 * q_exact
 * **********************************************/
/* define the source term on the right hand side */
/* exact solution */
double u_exact(const Vector & x){
	if(x.Size() == 2){
		double r(x(0) );
		double z(x(1) );

		if(sol_opt == 0){
			return  0.1*sin(M_PI*r) * cos(M_PI*z);
		}
		else if(sol_opt == 1){
			double d1 =  0.075385029660066;
			double d2 = -0.206294962187880;
			double d3 = -0.031433707280533;

			return   1./8.* pow(x(0),4)
				   + d1
				   + d2 * x(0)*x(0)
				   + d3 * ( pow(x(0),4) - 4. * x(0)*x(0) * x(1)*x(1) );
		}
		else if(sol_opt == 2){
			double d1 =  0.015379895031306;
    		double d2 = -0.322620578214426;
    		double d3 = -0.024707604384971;

			return   1./8.* pow(x(0),4)
				   + d1
				   + d2 * x(0)*x(0)
				   + d3 * ( pow(x(0),4) - 4. * x(0)*x(0) * x(1)*x(1) );
		}
		else if(sol_opt == 3){
			double r0 = -0.5, kr = 1.15*M_PI, kz = 1.15;
			return 
					sin( kr * (r+r0) ) * cos( kz*z);
		}
		else if(sol_opt == 4){
			return 0.25;
		}
		else if(sol_opt == 5){
			return 0.;
		}
		else{
			return 0;
		}	
	}
	else{
		return 0;
	}

}

/* exact q = - 1/r grad u */
void q_exact(const Vector & x,Vector & q){
	if(x.Size() == 2){
		 double r(x(0) );
		 double z(x(1) );

		 if(sol_opt == 0){
			 q(0) = -M_PI/r * cos(M_PI*r) * cos(M_PI*z);
			 q(1) =  M_PI/r * sin(M_PI*r) * sin(M_PI*z);
			 q *= 0.1;
			 
		 }
		 else if(sol_opt ==1){
			double d1 =  0.075385029660066;
			double d2 = -0.206294962187880;
			double d3 = -0.031433707280533;

			q(0) = -1./2. * pow( x(0),2 )
				   -d2*2.
				   -d3*( 4.* pow(x(0),2) - 8.* x(1)*x(1) ); 
			q(1) = -d3*( -8.* x(0) * x(1) );
		 }
		 else if(sol_opt ==2){
			double d1 =  0.015379895031306;
    		double d2 = -0.322620578214426;
    		double d3 = -0.024707604384971;

			q(0) = -1./2. * pow( x(0),2 )
				   -d2*2.
				   -d3*( 4.* pow(x(0),2) - 8.* x(1)*x(1) ); 
			q(1) = -d3*( -8.* x(0) * x(1) );
		 }
		 else if(sol_opt == 3){
			double r0 = -0.5, kr =1.15*M_PI, kz = 1.15;
			
			q(0) = - kr/r * cos( kr*(r+r0) ) * cos(kz*z);
			q(1) =   kz/r * sin( kr*(r+r0) ) * sin(kz*z);
		 }
		 else{
			q = 0.;
		 }
	}
	else{
		q  = 0.;
	}
}

/********************************
 * linear source term
 * ******************************/
double linear_source(const Vector & x){
	if(x.Size() == 2){
		 double r(x(0) );
		 double z(x(1) );

		 if(sol_opt == 0){
			 double u = u_exact(x);
			 return 0.1*
				   (
					 2./r * M_PI*M_PI * sin(M_PI*r) * cos(M_PI*z)
				    +1./r/r * M_PI * cos(M_PI*r) * cos(M_PI*z)
					) 
				    -u + u * u + exp(-u)
//				    -u_exact(x) - u_exact(x) * u_exact(x)
//				    -u_exact(x) + u_exact(x) * u_exact(x)
					;
//				    -u_exact(x) + 0.01 * u_exact(x) * u_exact(x); 
		 }
		 else if( (sol_opt == 1) || (sol_opt == 2) ){
			return -x(0) 
				   -u_exact(x) + 0.5*u_exact(x)*u_exact(x); 
		 }
		 else if( sol_opt == 3){
			double r0 = -0.5, kr = 1.15*M_PI, kz = 1.15;
			double u = u_exact(x);
			return  
				  kr/r/r * cos( kr*(r+r0) ) * cos(kz * z)
				  + u*u + exp(-u);
		 }
		 else if(sol_opt == 4){
			return 0;
		 }
		 else{
			return 0;
		 }
	}
	else{
		return 0;
	}
}


