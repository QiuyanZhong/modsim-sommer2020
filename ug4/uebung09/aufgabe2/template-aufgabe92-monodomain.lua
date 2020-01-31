-- Copyright (c) 2010-2016:  G-CSC, Goethe University Frankfurt
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
local gridName	= util.GetParam("--grid", "laplace_sample_grid_2d.ugx", --WAS 2D
							"filename of underlying grid")
local numRefs		= util.GetParamNumber("--numRefs", 4, "number of refinements")

util.CheckAndPrintHelp("Beispiel: Mono-Domain-Gleichungen (2D)");


-- initialize ug with the world dimension dim=2 and the algebra type
local blockSize = 2 -- select 2 for 2x2 point-block system
InitUG(2, AlgebraType("CPU", blockSize));  


-- Load a domain without initial refinements.
local mandatorySubsets = {"Inner", "Boundary"}
dom = util.CreateDomain(gridName, 0, mandatorySubsets)

-- Refine the domain (redistribution is handled internally for parallel runs)
print("refining...")
util.refinement.CreateRegularHierarchy(dom, numRefs, true)

-----------------------------------------
-- A) Modellparameter
-- 
-- \chi (Cm \frac{\partial V}{\partial t} + I)  + \nabla \cdot [ -sigma \nabla V] = 0
-- Transformation: u = (V-Vr)/(Vp-Vr)   <->  V = (Vp-Vr)*u + Vr
-- 
-- Einheiten:
-- 
-- L = 1 cm
-- T = 1 ms
-- U = 1 mV
-- 
-----------------------------------------

local Vr = -85   -- mV
local Vs = -75   -- mV
local Vp =  15   -- mV

local scaleV = (Vp-Vr)        -- Skalierung [mV]
local a = (Vs-Vr)/(Vp-Vr)     -- Anregeschwellwert [1]


local VoltPerMilliVolt = 1e-3
local Cm  = 1.0 * VoltPerMilliVolt --  uF/cm^2,  where: 1 F=C/V = As / V => 1 mF = ms A/V  => 1 uF = ms * mA /V

local c1 = 0.175 -- ms^{-1}
local c2 = 0.03  -- ms^{-1}
local c3 = 0.011 -- ms^{-1}
local  b = 0.55  -- [1]


local sigma = 2.0  * VoltPerMilliVolt  -- mS/cm, 1 [S = A/V => 1 mS = 1 mA / V]
local chi = 2000 -- cm^{-1}

local DDiff = sigma/(Cm*chi)     --    mA/(V*cm) * V/(mA ms) / cm^{-1} = cm^2 / ms 
local lchar = 1.0 -- characteristic length scale for diffusion
local tDiff = lchar*lchar/DDiff


print ("Activation a:"..a)
print ("ScaleV :"..scaleV)


print ("DDiff = "..DDiff.." cm^2/ms")
print ("tDiff = "..tDiff.." ms")
print ("tReact = "..1/c1.." ms")
print ("tReact = "..1/c2.." ms")
print ("tReact = "..1/b.." ms")
print ("tReactDamp = "..1/(a*c1+c3*b).." ms")

local uinfty = 0.0
local winfty = uinfty / b


-----------------------------------------
-- B) Ansatzraum
-----------------------------------------


local approxSpace = ApproximationSpace(dom)
-- TODO: Define approximation space containing u,w (with 1st order Lagrange) 

print("approximation space:")
approxSpace:init_levels()
approxSpace:init_top_surface()
approxSpace:print_statistic()


-----------------------------------------
-- C) Elementdiskretisierung (FV)
-----------------------------------------
local elemDisc = {}
-- TODO: Define elemDisc["u"], elemDisc["w"]


-- elemDisc["u"]:set_mass_scale(1.0)  -- default
-- elemDisc["w"]:set_mass_scale(1.0)  -- default

-----------------------------------------
-- C).1 Diffusionstensor
-- Df = sigma/(Cm*chi)  
-----------------------------------------
local diffTensorU = DDiff  -- for task 2a)
-- TODO: Replace 'diffTensorU' by matrix object for task 2b)

elemDisc["u"]:set_diffusion(diffTensorU)
elemDisc["w"]:set_diffusion(0.0)

-----------------------------------------
-- C).2 Reaktionsterme:
-- \dot u = (c1*u*(u-a)*(1.0-u) - c2*w)
-- \dot w = b*(v - d*w)
-----------------------------------------


function ReactionU(u,w) 
  return -- TODO: implement 
end

function ReactionU_dU(u,w) 
    return -- TODO: implement 
end

function ReactionU_dW(u,w)  
  return -- TODO: implement 
end

function ReactionW(u,w) 
  return -- TODO: implement 
end

function ReactionW_dU(u,w) 
    return -- TODO: implement 
end

function ReactionW_dW(u,w)  
  return -- TODO: implement 
end



local nonlinearGrowth = {}
nonlinearGrowth["u"] = LuaUserFunctionNumber("ReactionU", 2)
nonlinearGrowth["u"]:set_input(0, elemDisc["u"]:value())
nonlinearGrowth["u"]:set_input(1, elemDisc["w"]:value())
nonlinearGrowth["u"]:set_deriv(0, "ReactionU_dU")
nonlinearGrowth["u"]:set_deriv(1, "ReactionU_dW")


-- TODO: nonlinearGrowth["w"] = ...


elemDisc["u"]:set_reaction(nonlinearGrowth["u"])          
elemDisc["w"]:set_reaction(nonlinearGrowth["w"])  

-----------------------------------------
--  C).3 Quellen = Externe Ströme (nicht benutzt!)
-- 1.0 / (Cm*scaleV)
-----------------------------------------
-- function ISourceU(x,y,t,si)  return 0.0/(Cm*scaleV) end
-- elemDisc["u"]:set_source("ISourceU") 


-----------------------------------------
-- D) Anfangswerte
-----------------------------------------
-- Initial values ("Anfangswerte")
function MyInitialValueU(x, y)
 return -- TODO: implement initial value for "u"
end

function MyInitialValueW(x, y)
  return winfty 
end


-----------------------------------------
-- E) Randwerte
-----------------------------------------
local dirichletBND = DirichletBoundary()
dirichletBND:add(uinfty, "u", "Boundary")
dirichletBND:add(winfty, "w", "Boundary")


-----------------------------------------
-- F) Diskretisierung auf ganzem Gebiet
-----------------------------------------
local domainDisc = DomainDiscretization(approxSpace)
domainDisc:add(elemDisc["u"])
domainDisc:add(elemDisc["w"])
domainDisc:add(dirichletBND)

-----------------------------------------
-- F) Aufsetzen des Lösers
--    (using 'util/solver_util.lua')
-----------------------------------------
local solverDesc = {
	
	-- Newton's method for non-linear problem
	type = "newton",

  -- BiCGStab 
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

-----------------------------------------
-- G) Lösen
-----------------------------------------

local nlsolver = util.solver.CreateSolver(solverDesc)
print (nlsolver)

print("\nsolving...")
local A = AssembledLinearOperator(domainDisc)
local u = GridFunction(approxSpace)
local b = GridFunction(approxSpace)
u:set(0.0)
-- domainDisc:adjust_solution(u)
-- domainDisc:assemble_linear(A, b)

Interpolate("MyInitialValueU", u, "u")
Interpolate("MyInitialValueW", u, "w")

local startTime = 0
local endTime = tDiff
local dt = (endTime-startTime)/200.0
util.SolveNonlinearTimeProblem(u, domainDisc, nlsolver, VTKOutput(),
"Aufgabe9_2", "ImplEuler", 1, startTime, endTime, dt);

print("done")
