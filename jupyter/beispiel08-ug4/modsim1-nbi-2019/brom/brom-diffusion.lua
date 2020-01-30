-- Copyright (c) 2010-2020:  G-CSC, Goethe University Frankfurt
-- Authors: Arne Naegel
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
local ARGS ={
 gridName	= util.GetParam("--grid", "grids/brom2.ugx",
							"filename of underlying grid"),
 numRefs		= util.GetParamNumber("--numRefs", 2, "number of refinements"),

 steadyState	= util.HasParamOption("--steadyState", "If specified, the steady state of the problem is computed. Else a time-dependent problem is computed."),

 endTime 	= util.GetParamNumber("--endTime", 0.4, "simulated time frame in seconds"),
 dt			= util.GetParamNumber("--dt", 0.0125, "time step size"),
}

-- initialize ug with the world dimension 3 and an algebra system with scalar coefficients
InitUG(3, AlgebraType("CPU", 1));


-- Load a domain without initial refinements.
local requiredSubsets = {"INNER", "WALL", "IN"}
local dom = util.CreateDomain(ARGS.gridName, 0, requiredSubsets)

-- Refine the domain (redistribution is handled internally for parallel runs)
print("refining...")
util.refinement.CreateRegularHierarchy(dom, ARGS.numRefs, true)


-- Create approximation space.
local approxSpaceDesc = { fct = "c", type = "Lagrange", order = 1 }

local approxSpace = ApproximationSpace(dom)
approxSpace:add_fct(approxSpaceDesc.fct, approxSpaceDesc.type, approxSpaceDesc.order)
approxSpace:init_levels()
approxSpace:init_top_surface()

print("Approximation space:")
approxSpace:print_statistic()


-- FE/FV discretization.
local elemDisc = ConvectionDiffusion("c", "INNER", "fv1")
elemDisc:set_diffusion(10)


local dirichletBnd = DirichletBoundary()
dirichletBnd:add(0.0, "c", "IN")
-- dirichletBnd:add(1.0, "c", "TOP")


local domainDisc = DomainDiscretization(approxSpace)
domainDisc:add(elemDisc)
domainDisc:add(dirichletBnd)



-- set up solver (using 'util/solver_util.lua')
local solverDesc = {
	type = "bicgstab",
	precond = {
		type		= "gmg",
		approxSpace	= approxSpace,
		smoother	= "ilu",
		baseSolver	= "lu"
	}
}

local solver = util.solver.CreateSolver(solverDesc)


if ARGS.steadyState then
	local A = AssembledLinearOperator(domainDisc)
	local b = GridFunction(approxSpace)
	domainDisc:adjust_solution(u)
	domainDisc:assemble_linear(A, b)

	solver:init(A, u)
	solver:apply(u, b)

	local solFileName = "sol_brom"
	print("writing solution to '" .. solFileName .. "'...")
	WriteGridFunctionToVTK(u, solFileName)
	SaveVectorForConnectionViewer(u, solFileName .. ".vec")
else

  function InitialValue(x,y,z,t,si) 
    if (z<1.75) then return 0.0 else return 1.0 end
  end


  print("\nsolving...")
  local u = GridFunction(approxSpace)
  u:set(0.0)
  Interpolate("InitialValue", u, "c")


	local startTime = 0
	util.SolveLinearTimeProblem(u, domainDisc, solver, VTKOutput(), "sol_brom",
								"ImplEuler", 1, startTime, ARGS.endTime, ARGS.dt); 
end

print("done")

