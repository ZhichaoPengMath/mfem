#include "mfem.hpp"

using namespace std;

namespace mfem
{
	/* **************************
	 * Coefficient for F(u),
	 * where F(u) is a scalar and 
	 * u is a scalar grid function
	 * **************************/
	class FUCoefficient : public Coefficient
	{
		private:
			GridFunction *u;
			double (*Function)(double uu); 
		public:
			FUCoefficient(GridFunction *_u, double (*_F)(double x) ): u(_u), Function(_F){};

			double Eval(ElementTransformation &T, const IntegrationPoint &ip);
	};
	double FUCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
	{
		/* coordinate */
		double x[3];
		Vector transip(x,3);
		T.Transform(ip, transip);

		double xi = transip(0);
		double yi = transip(1);

		double uip = u->GetValue(T.ElementNo,ip);

		return (*Function)(uip);

	}
	/* **************************
	 * Coefficient for F(u),
	 * where F(u,x) is a scalar and 
	 * u is a scalar grid function
	 * **************************/
	class FUXCoefficient : public Coefficient
	{
		private:
			GridFunction *u;
			double (*Function)(double uu, const Vector & xx); 
		public:
			FUXCoefficient(GridFunction *_u, double (*_F)(double uu, const Vector & xx) ): u(_u), Function(_F){};

			double Eval(ElementTransformation &T, const IntegrationPoint &ip);
	};
	double FUXCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
	{
		/* coordinate */
		double x[3];
		Vector transip(x,3);
		T.Transform(ip, transip);

		double xi = transip(0);
		double yi = transip(1);

		double uip = u->GetValue(T.ElementNo,ip);

		return (*Function)(uip,transip);

	}
/************************************************************/
	class RHSCoefficient : public Coefficient
	{
		private:
			GridFunction *u;
		public:
			RHSCoefficient(GridFunction *_u): u(_u){};

			double Eval(ElementTransformation &T, const IntegrationPoint &ip);
	};

	double RHSCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
	{
		/* parameters */
		double kr  = 1.15*M_PI, kz = 1.15;
		double r0  = -0.5;


		/* coordinate */
		double x[3];
		Vector transip(x,3);
		T.Transform(ip, transip);

		double xi = transip(0);
		double yi = transip(1);

		double uip = u->GetValue(T.ElementNo,ip);

		return uip - 0.5*uip*uip;
		return - 0.5*exp(-uip);
		return uip - exp(-uip);
		return uip;
//		return exp(-uip);
//		return  uip - xi * xi * (sin(4*M_PI*xi) + sin(4*M_PI*yi) + yi )
//				-12. *M_PI * cos(4.*M_PI * xi) 
//		        +xi * 16. *M_PI*M_PI * sin(4.*M_PI * xi)
//				+xi * 16. *M_PI*M_PI * sin(4.*M_PI * yi);
	}

/************************************************************/
	class DFDUCoefficient : public Coefficient
	{
		private:
			GridFunction *u;
		public:
			DFDUCoefficient(GridFunction *_u):u(_u){};

			double Eval(ElementTransformation &T, const IntegrationPoint &ip);
	};

	double DFDUCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
	{
		/* parameters */
		double kr  = 1.15*M_PI, kz = 1.15;
		double r0  = -0.5;

		/* coordinate */
		double x[3];
		Vector transip(x,3);
		T.Transform(ip, transip);

		double xi = transip(0);
		double yi = transip(1);

		double uip = u->GetValue(T.ElementNo,ip);

		return 1 - uip;
//		return 0.5*exp(-uip);
//		return 1+exp(-uip);
//		return 1+0.1*exp(-uip);
//		return exp(-uip);
		return 1;
	}

}
