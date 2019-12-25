1 anderson_dpgp_gs.cpp 
  anderson_reduced_system_operator.hpp
  SourceAndBoundary.hpp
  RHSCoefficient.hpp
  rc_anderson

Anderson mixing method to solve nonlinear Grad-Shafranov equation

2 amr_anderson_dpgp_gs.cpp 
  amr_anderson_reduced_system_operator.hpp
  SourceAndBoundary.hpp
  RHSCoefficient.hpp
  rc_anderson

Anderson mixing method to solve nonlinear Grad-Shafranov equation, with conforming
adaptive mesh refinement

2 nonlinear_dpgp_gs.cpp 
  nonlinear_reduced_system_operator.hpp
  SourceAndBoundary.hpp
  RHSCoefficient.hpp
  rc_inexact_newton
  rc_dpg_S2hypre

JFNK solving nonlinear Grad-Shafranov equation, not working well for many problems,
as the preconditioner for Jacobian matrix is not good enough

