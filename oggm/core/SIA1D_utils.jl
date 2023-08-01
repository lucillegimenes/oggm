
#For differential computation in the numerical scheme
diff_x!(O, I, Δx) = @views @. O = (I[2:end] - I[1:end - 1])./ Δx
avg!(O, I) = @views @. O = (I[2:end] + I[1:end - 1])./2.0

include("SIA1D.jl")

"""
SIA1D!(dH, H, SIA1Dmodel,t)

Compute an in-place step of the Shallow Ice Approximation PDE in a forward model
"""

function SIA1D!(dH::Vector{Float64}, H::Vector{Float64}, iceflow_model::SIA1Dmodel, t::Float64)
    # Retrieve model parameters
    Δx::Int64 = iceflow_model.Δx
    Γ::Float64 = iceflow_model.Γ
    Γs::Float64 = iceflow_model.Γs
    dSdx::Vector{Float64}= iceflow_model.dSdx
    ∇Sx::Vector{Float64} = iceflow_model.∇Sx
    w::Vector{Float64} = iceflow_model.w 
    w̅ ::Vector{Float64} = iceflow_model.w̅
    w0::Vector{Float64} = iceflow_model.w0
    w0_stag::Vector{Float64} = iceflow_model.w0_stag
    λ::Vector{Float64} = iceflow_model.λ
    S::Vector{Float64} = iceflow_model.S
    B::Vector{Float64} = iceflow_model.B
    H̅::Vector{Float64} = iceflow_model.H̅
    D::Vector{Float64} = iceflow_model.D
    Fx::Vector{Float64} = iceflow_model.Fx 
    Fxx::Vector{Float64} = iceflow_model.Fxx


    # First, enforce values to be positive
    map!(x -> ifelse(x>0.0,x,0.0), H, H)

    @. S = B + H
    @. w = w0 + (λ* H)

    #Staggered thickness
    avg!(H̅,H)

    #Staggered width
    avg!(w̅,w)

    #(Staggered) slope
    diff_x!(dSdx,S,Δx)

    # Diffusivity
    @views @. D[2:end-1] = ((dSdx^2.0)^((n-1)/2.0)) * ((w̅ + w0_stag)/2.0) * (Γ * H̅^(n.+2) + Γs*(H̅^n))

    append!(S,S[end])
    prepend!(S,S[1])

    diff_x!(∇Sx,S,Δx)

    @. Fx = -D * ∇Sx
    diff_x!(Fxx,Fx,Δx)

    dH .=  .-Fxx./w

    # Clip negative ice thickness values
    #@views H[H.<0.0] .= 0.0
    @assert H[end] .< 10.0 "Glacier exceeding boundaries! at time $(t/sec_in_year)"

    popfirst!(S)  #Useful for calculation at the beginning of the function 
    pop!(S)

end

"""
glacier_evolution_store(SIA1D, solver, reltol, y0, y1, l_var, do_geom, do_fl_diag, mb_step, dyn_spin_thick)

Runs the flowline dynamics up to given year date y1
This function runs the model for the time difference y1-self.y0 and stores diagnostic variables to give back to OGGM
"""

function glacier_evolution_store(;SIA1D::Py, 
                                solver, 
                                reltol::Float64,
                                y0::Int64, 
                                y1::Int64, 
                                l_var::PyList{Any}, 
                                do_geom::Int64, 
                                do_fl_diag::Int64,
                                mb_step::String,
                                dyn_spin_thick::Float64)

    # Initialize SIA1D model with variables from current flowline 
    iceflow_model::SIA1Dmodel = SIA1Dmodel()
    initialize_iceflow_model!(iceflow_model, SIA1D)

    #Time parameters 

    sim_params::SimulationParameters = SimulationParameters()
    initialize_sim_params!(sim_params, SIA1D, y0, y1, mb_step)


    #### Initialize necessary storage structs     
    #Diagnostic
    diag_jl::diag = diag(sim_params.nₜm,sim_params.fact_m,sim_params.diff, SIA1D,l_var,dyn_spin_thick)

    ##Geometry and flowline diagnostic
    do_geom = Bool(do_geom)
    do_fl_diag = Bool(do_fl_diag)
    is_tidewater = pyconvert(Bool,SIA1D.is_tidewater)
    geom_var::geom, fl_var::fl_diag = geom_fl(do_geom,do_fl_diag,sim_params.nₜ,iceflow_model.nx,SIA1D,sim_params.diff,is_tidewater)


    #Callback functions for the mass balance and update the SIA1D object 
    tstops, _ = define_callback_steps(sim_params.tspan)
    stop_condition(u,t,integrator) = stop_condition_tstops(u,t,integrator, tstops) #closure

    #To be called during the callbacks
    function action!(integrator)
        #Adding the mass balance at each time stops
        sim_params.yr = sim_params.y₀slf + ((integrator.t - sec_in_month)/sec_in_year)
        
        get_mb!(iceflow_model.mb, iceflow_model.B .+ integrator.u ,sim_params.yr,SIA1D)
        integrator.u .+= (iceflow_model.mb .* sec_in_month)
        integrator.u[integrator.u.<0.0] .= 0.0

        #writing the thickness solution every year as a property of the SIA1D object
        setproperty!(SIA1D, :t, integrator.t)
        setproperty!(SIA1D.fls[0], :thick, np.array(integrator.u))

        #Yearly Index
        tᵢ = floor(Int,integrator.t ./ sec_in_year) + 1 
        #Monthly index
        tᵢm = floor(Int,(integrator.t ./sec_in_year)./(1/12)) + 1

        if sim_params.yr >= sim_params.y₀
            if sim_params.mb_step == "annual" 
                if tᵢ > 1 && tᵢm%12 == 1 #first year is already saved
                    tᵢ = tᵢ + sim_params.diff
                    #Writing the diagnostic variables
                    update_diag!(diag_jl,SIA1D,l_var,dyn_spin_thick,tᵢ)    

                    #Writing for the geometry (and flowline diagnostic)
                    if do_geom || do_fl_diag
                        update_geom!(geom_var,SIA1D,is_tidewater,tᵢ)
                        if do_fl_diag
                            update_fl_diag!(fl_var,SIA1D,tᵢ)
                        end 
                    end 
                end 
            else #store monthly steps
                #Writing the diagnostic variables
                update_diag!(diag_jl,SIA1D,l_var,dyn_spin_thick,tᵢm+(sim_params.diff*12))
                if tᵢm%12 == 1
                    tᵢ = tᵢ + sim_params.diff
                    #Writing for the geometry (and flowline diagnostic) only once a year 
                    if do_geom || do_fl_diag
                        update_geom!(geom_var,SIA1D,is_tidewater,tᵢ)
                        if do_fl_diag
                            update_fl_diag!(fl_var,SIA1D,tᵢ)
                        end 
                    end    
                end 

            end 
        end 

    end   

    #Defining the callback
    cb_MB = DiscreteCallback(stop_condition, action!)

    #Solving the problem         
    iceflow_prob = ODEProblem(SIA1D!,iceflow_model.H,sim_params.tspan,tstops=tstops,iceflow_model)
    iceflow_sol = solve(iceflow_prob,solver,callback=cb_MB, tstops=tstops, reltol=reltol,save_everystep=false,dense=false)

    return SIA1D,diag_jl, geom_var, fl_var
    
end 



#Basically like glacier_evolution_store but without storing anything
function glacier_evolution(;
    SIA1D::Py, #IceflowJuliaModel object
    solver, #Solver to use
    reltol::Float64, #Relative tolerance
    y0::Int64, #starting year of the simulation
    y1::Int64) #ending year of the simulation

    mb_step::String ="annual"

    # Initialize SIA1D model with variables from current flowline 
    iceflow_model::SIA1Dmodel = SIA1Dmodel()
    initialize_iceflow_model!(iceflow_model, SIA1D)
    
    #Time parameters 
    
    sim_params::SimulationParameters = SimulationParameters()
    initialize_sim_params!(sim_params, SIA1D, y0, y1, mb_step)


    #Callback functions for the mass balance and update the SIA1D object 
    tstops, _ = define_callback_steps(sim_params.tspan)
    stop_condition(u,t,integrator) = stop_condition_tstops(u,t,integrator, tstops) #closure

    #To be called during the callbacks
    function action!(integrator)
        sim_params.yr = sim_params.y₀slf + ((integrator.t - sec_in_month)/sec_in_year)
        get_mb!(iceflow_model.mb, iceflow_model.B .+ integrator.u, sim_params.yr, SIA1D)
        integrator.u .+= (iceflow_model.mb .* sec_in_month)
        integrator.u[integrator.u.<0.0] .= 0.0

        #writing the thickness solution every year as a property of the SIA1D object
        setproperty!(SIA1D, :t, integrator.t)
        setproperty!(SIA1D.fls[0], :thick, np.array(integrator.u))

    end   

    #Defining the callback
    cb_MB = DiscreteCallback(stop_condition, action!)

    #Solving the problem         
    iceflow_prob = ODEProblem(SIA1D!,iceflow_model.H,sim_params.tspan,tstops=tstops,iceflow_model)
    iceflow_sol = solve(iceflow_prob,solver,callback=cb_MB, tstops=tstops, reltol=reltol,save_everystep=false,dense=false)

    return SIA1D

end




##### CALLBACK FUNCTIONS #####

function get_mb!(mb::Vector{Float64}, heights::Vector{Float64},y::Float64,SIA1D::Py)
    mb .= pyconvert(Vector{Float64},SIA1D.get_mb(heights,y,fl_id=0))     
end

function define_callback_steps(tspan::Tuple)
    step = sec_in_month
    tmin_int = Int(tspan[1])
    tmax_int = Int(tspan[2])+1
    tstops = range(tmin_int+step, tmax_int, step=step) |> collect
    tstops = filter(x->( (Int(tspan[1])<x) & (x<=Int(tspan[2])) ), tstops)
    return tstops, step
end

function stop_condition_tstops(u,t,integrator, tstops) 
    t in tstops
end


##############################