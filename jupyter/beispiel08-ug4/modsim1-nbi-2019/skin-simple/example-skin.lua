-- Copyright (c) 2010-2016:  G-CSC, Goethe University Frankfurt
-- Authors: Andreas Vogel, Sebastian Reiter
-- 
-- This file is part of UG4.
-- 
-- UG4 is free software: you can redistribute it and/or modify it under the
-- terms of the GNU Lesser General Public License version 3 (as published by the
-- Free Software Foundation) with the following additional attribution
-- requirements (according to LGPL/GPL v3 §7):
-- 
-- (1) The following notice must be displayed in the Appropriate Legal Notices
-- of covered and combined works: "Based on UG4 (www.ug4.org/license)".
-- 
-- (2) The following notice must be displayed at a prominent place in the
-- terminal output of covered works: "Based on UG4 (www.ug4.org/license)".
-- 
-- (3) The following bibliography is recommended for citation and must be
-- preserved in all covered files:
-- "Reiter, S., Vogel, A., Heppner, I., Rupp, M., and Wittum, G. A massively
--   parallel geometric multigrid solver on hierarchically distributed grids.
--   Computing and visualization in science 16, 4 (2013), 151-164"
-- "Vogel, A., Reiter, S., Rupp, M., Nägel, A., and Wittum, G. UG4 -- a novel
--   flexible software system for simulating pde based models on high performance
--   computers. Computing and visualization in science 16, 4 (2013), 165-179"
-- 
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
-- GNU Lesser General Public License for more details.


-- Load utility scripts (e.g. from from ugcore/scripts)
ug_load_script("ug_util.lua")
ug_load_script("util/refinement_util.lua")

-- Parse parameters and print help
local ARGS = {
 gridName	= util.GetParam("--grid", "simple-skin.ugx",
							"filename of underlying grid"),
 numRefs		= util.GetParamNumber("--numRefs", 3, "number of refinements")
}
util.CheckAndPrintHelp("Waermeleitungsgleichung mit nicht-linear(in 2D)");


-- initialize ug with the world dimension dim=2 and the algebra type
InitUG(2, AlgebraType("CPU", 1));


-- Load a domain without initial refinements.
local mandatorySubsets = {"INNER", "TOP", "BOT"}
dom = util.CreateDomain(ARGS.gridName, 0, mandatorySubsets)

-- Refine the domain (redistribution is handled internally for parallel runs)
print("refining...")
util.refinement.CreateRegularHierarchy(dom, ARGS.numRefs, true)


-- set up approximation space: linear functions
local approxSpace = ApproximationSpace(dom)
approxSpace:add_fct("u", "Lagrange", 1)
approxSpace:init_levels()
approxSpace:init_top_surface()

print("approximation space:")
approxSpace:print_statistic()

-- Initial values ("Anfangswerte")
-- start with c=1 in circle around center (and 0 elsewhere) 
function MyInitialValue(x, y)
  return 0.0 
end

--[[ set up discretization for 
  $$ \frac{\partial Ku}{\partial t} - KD \triangle c = 0$$
--]]



local membraneDesc ={
  { subset = "INNER", D=1.0, K=1.0},
  { subset = "INNER_COR", D=0.1, K=1.0},
}

local K = membraneDesc[1].K
local D = membraneDesc[1].D

local MyCharTime = 4.0/membraneDesc[1].D


-- Finite Volumen-Verfahren 
local elemDisc ={}
elemDisc[1]= ConvectionDiffusion("u", membraneDesc[1].subset, "fv1")
elemDisc[1]:set_diffusion(membraneDesc[1].K*membraneDesc[1].D)
elemDisc[1]:set_mass_scale(membraneDesc[1].K)

--[[ Optional fuer simple-skin-with-cor.ugx
  
elemDisc[2]= ConvectionDiffusion("u", membraneDesc[2].subset, "fv1")
elemDisc[2]:set_diffusion(membraneDesc[2].K*membraneDesc[2].D)
elemDisc[2]:set_mass_scale(membraneDesc[2].K)
-- ]]

local dirichletBND = DirichletBoundary()
dirichletBND:add(0.0, "u", "BOT")
dirichletBND:add(1.0, "u", "TOP")

local domainDisc = DomainDiscretization(approxSpace)
domainDisc:add(elemDisc[1])
--domainDisc:add(elemDisc[2])
domainDisc:add(dirichletBND)


-- set up solver (using 'util/solver_util.lua')
local solverDesc = {
	
	type = "newton",
	
	linSolver = {
	 type = "bicgstab",
	
	
	 precond = {
		  type		= "gmg",
		  approxSpace	= approxSpace,
		  smoother	= "sgs",
		  baseSolver	= "lu"
	 },
	
  },
	
}

out = VTKOutput()




local tOld = 0.0
local jOld = 0.0
local mOld = 0.0

function MyPostProcess(u, step, time)
  
  -- 1) print solution to file
  out:print("SkinTransient", u, step, time)
  
  -- 2) compute fluxes
  local value={}
  value["BOT"] = IntegrateNormalGradientOnManifold(u, "u", "BOT", "INNER")
  value["TOP"] = IntegrateNormalGradientOnManifold(u, "u", "TOP", "INNER")
  
  local jTOP = value["TOP"]*K*D
  local jBOT = value["BOT"]*K*D
  print ("flux_top (\t"..time.."\t)=\t"..jTOP)
  print ("flux_bot (\t"..time.."\t)=\t"..jBOT)
  
  -- 3) compute mass
  local dt = time - tOld
  local mass = mOld + (time - tOld)/2.0*(jBOT + jOld)
  print ("mass_bot (\t"..time.."\t)=\t"..mass)
  
  -- 4) compute lag time
  print ("tlag=".. time - mass/jBOT )
  
  -- 5) updates
  tOld = time
  jOld = jBOT
  mOld = mass

  
end

local nlsolver = util.solver.CreateSolver(solverDesc)
print (nlsolver)


-- Transientes Problem
print("\Stationaeres Problem:")
local A = AssembledLinearOperator(domainDisc)
local u = GridFunction(approxSpace)
local b = GridFunction(approxSpace)
u:set(0.0)
domainDisc:adjust_solution(u)
domainDisc:assemble_linear(A, b)





-- Transientes Problem
print("\Transientes Problem:")
Interpolate("MyInitialValue", u, "u")
local startTime = 0
local endTime = 10*MyCharTime
local dt = (endTime-startTime)/100.0
util.SolveNonlinearTimeProblem(u, domainDisc, nlsolver, MyPostProcess,
"u_skin_transient", "ImplEuler", 1, startTime, endTime, dt);


print("done")
