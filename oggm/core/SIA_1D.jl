xr=pyimport("xarray")
np=pyimport("numpy")

#Constants definition
const sec_in_day = 24.0 * 60.0 * 60.0
const sec_in_year = sec_in_day * 365.0
const sec_in_month = 2628000.0
const glen_n = 3.0
const ρ = 900.0
const g = 9.81

#For differential computation in the numerical scheme
diff1!(O, I, dx) = @views @. O[1:end] = (I[2:end] - I[1:end - 1])./ dx
#diff2!(O, I, dx) = @views @. O[2:end-1] = (I[3:end] - I[1:end - 2]) / dx
avg!(O, I) = @views @. O[1:end] = (I[2:end] + I[1:end - 1])./2.0

#Structs used to store variables
mutable struct diag
    volume_m3::Vector{Float64}
    area_m2::Vector{Float64}
    length_m::Vector{Float64}
    calving_m3::Vector{Float64}
    calving_rate_myr::Vector{Float64}
    volume_bsl_m3::Vector{Float64}
    volume_bwl_m3::Vector{Float64}
    area_m2_min_h::Vector{Float64}
end

mutable struct geom
    sects::Vector{Matrix{Float64}}
    widths::Vector{Matrix{Float64}}
    buckets::Vector{Vector{Float64}}
end 

mutable struct fl_diag
    thick_fl::Vector{Matrix{Float64}} 
    volume_bsl_fl::Vector{Vector{Float64}} 
    volume_bwl_fl::Vector{Vector{Float64}}
end 


function glacier_evolution_store(;SIA_1D::Py, 
                                solver, 
                                reltol::Float64,
                                y0::Int64, 
                                y1::Int64, 
                                l_var::PyList{Any}, 
                                do_geom::Int64, 
                                do_fl_diag::Int64,
                                mb_step::String,
                                dyn_spin_thick::Float64)
    #=
    Runs the flowline dynamics up to given year date y1
    This function runs the model for the time difference y1-self.y0 and stores diagnostic variables to give back to OGGM
    
    Parameters
    ----------
    SIA_1D: PyObject
        the SIA_1D(IceflowJuliaModel) object from OGGM
    solver: Julia solver
    reltol: Float64
        relative tolerance specified for the solver
    y0: Int64
        starting year for the diagnostic variables (can be different from SIA_1D.y0)
    y1: Int64
        upper time span for how long the model should run 
    l_var: PyList{Any}
        list of diagnostic variables to be stored
    do_geom: Int64 (0 or 1)
        to store model geometry or not during the simulation
    do_fl_diag: Int64 (0 or 1)
        to store flowline diagnostic variables or not during the simulation
    mb_step: String ('annual' or 'monthly')
        the timestep for diagnostic variables
    dyn_spin_thick: None or Float64
        if set to float (during dynamic spinup), used for the calculation of diagnostic variable 'area_min_h'
    =# 

    # Read out variables from current flowline 
    context = build_context(SIA_1D,y0,y1,mb_step)
    nx::Int64 = context[1]
    dx::Int64 = context[2]

    #Thickness
    H₀::Vector{Float64} = context[3]
    H_stag::Vector{Float64} = context[4]
    
    #Working with trapezoidal bed shape 
    width::Vector{Float64} = context[5]
    width_stag::Vector{Float64} = context[6]
    w₀::Vector{Float64} = context[7]
    w₀_stag::Vector{Float64} = context[8]
    λ::Vector{Float64} = context[9]

    bed_h::Vector{Float64}= context[10]

    #Pre-allocation
    surface_h::Vector{Float64} = context[11]
    slope::Vector{Float64} = context[12]
    slope_stag::Vector{Float64} = context[13]
    grad_x_diff::Vector{Float64} = context[14]
    flux_div::Vector{Float64} = context[15]
    diffusivity::Vector{Float64} = context[16]
    mb::Vector{Float64} = context[17]

    #Parameters for later computation
    A::Float64 = context[18]
    ρ_g::Float64 = context[19]
    Γ::Float64 = context[20]
    fs::Float64 = context[21]

    #Time parameters
    y₀::Int64 = context[22]
    y₀slf::Int64 = context[23]
    y₁::Int64 = context[24]

    tspan::Tuple = context[25]
    nₜ::Int64 = context[26]
    nₜm::Int64 = context[27]

    diff::Int64 = context[28]
    fac::Int64 = context[29]

    #To be passed as parameters to the iceflow! function
    p::Tuple = (dx, Γ, slope, slope_stag, width, width_stag, w₀, w₀_stag, λ, surface_h, bed_h, H_stag, grad_x_diff, flux_div, diffusivity, fs, ρ_g)

    ## Diagnostic         
    #Storage for diagnostic variables 
    diag_jl::diag = create_diag(nₜm,fac,diff, SIA_1D,l_var,dyn_spin_thick)

    ##Geometry and flowline diagnostic
    do_geom = Bool(do_geom)
    do_fl_diag = Bool(do_fl_diag)
    
    geom_var::geom, fl_var::fl_diag = create_geom_fl(do_geom,do_fl_diag,nₜ,nx,SIA_1D,diff)

    #Callback functions for the mass balance and update the SIA_1D object 
    tstops, _ = define_callback_steps(tspan)
    stop_condition(u,t,integrator) = stop_condition_tstops(u,t,integrator, tstops) #closure

    #To be called during the callbacks
    function action!(integrator)
        get_mb!(mb, bed_h .+ integrator.u ,SIA_1D.yr,SIA_1D)
        integrator.u .+= (mb .* sec_in_month)
        integrator.u[integrator.u.<0.0] .= 0.0

        #writing the thickness solution every year as a property of the SIA_1D object
        setproperty!(SIA_1D, :t, integrator.t)
        setproperty!(SIA_1D.fls[0], :thick, np.array(integrator.u))

        #Yearly Index
        tᵢ = floor(Int,integrator.t ./ sec_in_year) + 1 
        #Monthly index
        tᵢm = floor(Int,(integrator.t ./sec_in_year)./(1/12)) + 1

        if pyconvert(Float64,SIA_1D.yr) >= y₀
            if mb_step == "annual" 
                if tᵢ > 1 && tᵢm%12 == 1 #first year is already saved
                    tᵢ = tᵢ + diff
                    #Writing the diagnostic variables
                    update_diag!(diag_jl,SIA_1D,l_var,dyn_spin_thick,tᵢ)    

                    #Writing for the geometry (and flowline diagnostic)
                    if do_geom || do_fl_diag
                        update_geom!(geom_var,SIA_1D,tᵢ)
                        if do_fl_diag
                            update_fl_diag!(fl_var,SIA_1D,tᵢ)
                        end 
                    end 
                end 
            else #store monthly steps
                #Writing the diagnostic variables
                update_diag!(diag_jl,SIA_1D,l_var,dyn_spin_thick,tᵢm+(diff*12))
                if tᵢm%12 == 1
                    tᵢ = tᵢ + diff
                    #Writing for the geometry (and flowline diagnostic) only once a year 
                    if do_geom || do_fl_diag
                        update_geom!(geom_var,SIA_1D,tᵢ)
                        if do_fl_diag
                            update_fl_diag!(fl_var,SIA_1D,tᵢ)
                        end 
                    end    
                end 

            end 
        end 

    end   

    #Defining the callback
    cb_MB = DiscreteCallback(stop_condition, action!)

    #Solving the problem         
    iceflow_prob = ODEProblem(iceflow!,H₀,tspan,tstops=tstops,p)
    iceflow_sol = solve(iceflow_prob,solver,callback=cb_MB, tstops=tstops, reltol=reltol,save_everystep=false,dense=false)

    return SIA_1D,diag_jl, geom_var, fl_var
    
end


#### FUNCTIONS FOR OGGM OUTPUTS ####

function create_diag(nₜm::Int64,fac::Int64,diff::Int64, SIA_1D::Py,l_var::PyList{Any},dyn_spin_thick::Float64)
    diag_jl = diag(Vector{Float64}(zeros(nₜm)),Vector{Float64}(zeros(nₜm)),Vector{Float64}(zeros(nₜm)),
                Vector{Float64}(zeros(nₜm)),Vector{Float64}(zeros(nₜm)),Vector{Float64}(zeros(nₜm)),
                Vector{Float64}(zeros(nₜm)),Vector{Float64}(zeros(nₜm)))
    if diff >= 0               
        #Diagnostic at year 0 (and more if there is a spinup) 
        for i in 1:(diff*fac)+1
            update_diag!(diag_jl,SIA_1D,l_var,dyn_spin_thick,i)
        end 
    end

    return diag_jl
end  


function update_diag!(diag_jl::diag,SIA_1D::Py,l_var::PyList{Any},dyn_spin_thick::Float64,j::Int)
    diag_jl.volume_m3[j] = pyconvert(Float64,SIA_1D.volume_m3)
    diag_jl.area_m2[j] = pyconvert(Float64,SIA_1D.area_m2)
    diag_jl.length_m[j] = pyconvert(Float64,SIA_1D.length_m)
    diag_jl.calving_m3[j] = pyconvert(Float64,SIA_1D.calving_m3_since_y0)
    diag_jl.calving_rate_myr[j] = pyconvert(Float64,SIA_1D.calving_rate_myr)
    diag_jl.volume_bsl_m3[j] = pyconvert(Float64,SIA_1D.volume_bsl_m3)
    diag_jl.volume_bwl_m3[j] = pyconvert(Float64,SIA_1D.volume_bwl_m3)
    if "area_min_h" in l_var
        diag_jl.area_m2_min_h[j] = pyconvert(Float64,sum([sum(fl.bin_area_m2[fl.thick > dyn_spin_thick]) for fl in SIA_1D.fls]))
    end 
end 

function create_geom_fl(do_geom::Bool,do_fl_diag::Bool,nₜ::Int64,nx::Int64,SIA_1D::Py,diff::Int64)
    geom_var = geom([zeros(Float64,nₜ,nx) for fl in SIA_1D.fls],[zeros(Float64,nₜ,nx) for fl in SIA_1D.fls],[zeros(Float64,nₜ) for fl in SIA_1D.fls])
    fl_var = fl_var= fl_diag([zeros(Float64,nₜ,nx) for fl in SIA_1D.fls],[zeros(Float64,nₜ) for fl in SIA_1D.fls],[zeros(Float64,nₜ) for fl in SIA_1D.fls])

    #Storage for geometry
    if do_geom || do_fl_diag
        if diff >= 0               
            for i in 1:diff+1
                update_geom!(geom_var,SIA_1D,i)
            end 
        end 

        if do_fl_diag
            if diff >= 0               
                for i in 1:diff+1
                    update_fl_diag!(fl_var,SIA_1D,i)
                end 
            end     

        end 
    end 

    return geom_var, fl_var
end 

function update_geom!(geom_var::geom,SIA_1D::Py,j::Int)
    for (s,w,b,fl) in zip(geom_var.sects,geom_var.widths,geom_var.buckets,SIA_1D.fls)
        @views s[j,:] .= pyconvert(Vector{Float64},fl.section)
        @views w[j,:] .= pyconvert(Vector{Float64},fl.widths_m)
        if pyconvert(Bool,SIA_1D.is_tidewater)
            try 
                b[j] = pyconvert(Float64,fl.calving_bucket_m3)
            catch 
                println("AttributeError")
            end 
        end
    end 
end 

function update_fl_diag!(fl_var::fl_diag,SIA_1D::Py,j::Int)
    for (t,vs,vw,fl) in zip(fl_var.thick_fl,fl_var.volume_bsl_fl,fl_var.volume_bwl_fl,SIA_1D.fls)
        @views t[j,:] .= pyconvert(Vector{Float64},fl.thick)
        vs[j] = pyconvert(Float64,fl.volume_bsl_m3)
        vw[j] = pyconvert(Float64,fl.volume_bwl_m3)
    end 
end 

#### ####

#### FONCTIONS FOR THE CALLBACKS ####

function get_mb!(mb::Vector{Float64}, heights::Vector{Float64},y::Py,SIA_1D::Py)
    mb .= pyconvert(Vector{Float64},SIA_1D.get_mb(heights,y,fl_id=0))     
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

#### ####

function build_context(SIA_1D::Py, y0::Int64, y1::Int64, mb_step::String)
    # Read out variables from current flowline 
    fl = SIA_1D.fls[0]

    dx::Int64 = pyconvert(Int64,fl.dx_meter)
    nx::Int64 = pyconvert(Int64,fl.nx)
    λ::Vector{Float64} = pyconvert(Vector{Float64},fl._lambdas)
    
    #Thickness
    H₀::Vector{Float64} = pyconvert(Vector{Float64},fl.thick)
    H_stag::Vector{Float64} = zeros(Float64,nx-1)
    
    #Working with trapezoidal bed shape 
    width::Vector{Float64} = zeros(Float64,nx)
    width_stag::Vector{Float64} = zeros(Float64,nx-1)
    w₀::Vector{Float64} = pyconvert(Vector{Float64},fl._w0_m)
    w₀_stag::Vector{Float64} = zeros(Float64,nx-1)
    avg!(w₀_stag,w₀)

    bed_h::Vector{Float64} = pyconvert(Vector{Float64},fl.bed_h)

    #Pre-allocation
    surface_h::Vector{Float64} = zeros(Float64,nx)
    slope::Vector{Float64} = zeros(Float64,nx-1)
    slope_stag::Vector{Float64} = zeros(Float64,nx+1)
    grad_x_diff::Vector{Float64} = zeros(Float64,nx+1)
    flux_div::Vector{Float64} = zeros(Float64,nx)
    diffusivity::Vector{Float64} = zeros(Float64,nx+1)
    mb::Vector{Float64} = zeros(Float64,nx)

    #Parameters for later computation
    A::Float64 = pyconvert(Float64,SIA_1D.glen_a)
    ρ_g::Float64 = (ρ .* g).^glen_n
    Γ::Float64 = ((2.0 * A * ρ_g )/(glen_n + 2.0))
    fs::Float64 = pyconvert(Float64,SIA_1D.fs)

    #Time parameters
    y₀::Int64 = pyconvert(Int64,y0) #Starting year of the diagnostic variable datasets
    y₀slf::Int64 = pyconvert(Int64,SIA_1D.y0) #Starting year for the actual simulation
    y₁::Int64 = pyconvert(Int64,y1)

    tspan::Tuple = (0.0 , (y₁ - y₀slf)*sec_in_year)
    nₜ::Int64 = Int(y1-y0) + 1 #Number of years for the output

    diff::Int64 = y₀slf-y₀ #This parameter is useful in case of spinup (i.e when y₀slf =/= y₀)

    #Required MB step
    if mb_step =="annual"
        nₜm = nₜ
        fac = 1
    else 
        nₜm = (nₜ-1)*12 +1
        fac = 12
    end 

    context = (nx, dx, H₀, H_stag, width, width_stag, w₀, w₀_stag, λ, bed_h, surface_h, slope, slope_stag, grad_x_diff, flux_div, diffusivity, mb,
                A, ρ_g, Γ, fs, y₀, y₀slf, y₁, tspan, nₜ, nₜm, diff, fac)
    
    return context 

end 


#Function with the numerical scheme for the SIA equation
function iceflow!(dH, H, p, t)
    # Retrieve model parameters
    dx::Int64,
    Γ::Float64,
    slope::Vector{Float64},
    slope_stag::Vector{Float64},
    width::Vector{Float64},
    width_stag::Vector{Float64},
    w₀::Vector{Float64},
    w₀_stag::Vector{Float64},
    λ::Vector{Float64},
    surface_h::Vector{Float64},
    bed_h::Vector{Float64},
    H_stag::Vector{Float64}, 
    grad_x_diff::Vector{Float64}, 
    flux_div::Vector{Float64},
    diffusivity::Vector{Float64},
    fs::Float64,
    ρ_g::Float64 = p

    # First, enforce values to be positive
    map!(x -> ifelse(x>0.0,x,0.0), H, H)

    @. surface_h = bed_h + H
    @. width = w₀ + (λ* H)

    #Staggered thickness
    avg!(H_stag,H)

    #Staggered width
    avg!(width_stag,width)    

    #(Staggered) slope
    diff1!(slope,surface_h,dx)

    # Diffusivity
    @views @. diffusivity[2:end-1] = ((slope^2.0)^((glen_n-1)/2.0)) * ((width_stag + w₀_stag)/2.0) * (Γ * H_stag^(glen_n.+2) + fs*ρ_g*(H_stag^glen_n))

    append!(surface_h,surface_h[end])
    prepend!(surface_h,surface_h[1])

    diff1!(slope_stag,surface_h,dx)

    @. grad_x_diff = -diffusivity * slope_stag
    diff1!(flux_div,grad_x_diff,dx)

    dH .=  .-flux_div./width 

    # Clip negative ice thickness values
    #@views H[H.<0.0] .= 0.0
    @assert H[end] .< 10.0 "Glacier exceeding boundaries! at time $(t/sec_in_year)"

    popfirst!(surface_h)  #Useful for calculation at the beginning of the function 
    pop!(surface_h)

end

#Basically like glacier_evolution_store but without storing anything
function glacier_evolution(;
    SIA_1D::Py, #IceflowJuliaModel object
    solver, #Solver to use
    reltol::Float64, #Relative tolerance
    y0::Int64, #starting year of the simulation
    y1::Int64) #ending year of the simulation

    mb_step:: String ="annual"
    context = build_context(SIA_1D,y0,y1,mb_step)
    nx::Int64 = context[1]
    dx::Int64 = context[2]

    #Thickness
    H₀::Vector{Float64} = context[3]
    H_stag::Vector{Float64} = context[4]
    
    #Working with trapezoidal bed shape 
    width::Vector{Float64} = context[5]
    width_stag::Vector{Float64} = context[6]
    w₀::Vector{Float64} = context[7]
    w₀_stag::Vector{Float64} = context[8]
    λ::Vector{Float64} = context[9]

    bed_h::Vector{Float64} = context[10]

    #Pre-allocation
    surface_h::Vector{Float64} = context[11]
    slope::Vector{Float64} = context[12]
    slope_stag::Vector{Float64} = context[13]
    grad_x_diff::Vector{Float64} = context[14]
    flux_div::Vector{Float64} = context[15]
    diffusivity::Vector{Float64} = context[16]
    mb::Vector{Float64} = context[17]

    #Parameters for later computation
    A::Float64 = context[18]
    ρ_g::Float64 = context[19]
    Γ::Float64 = context[20]
    fs::Float64 = context[21]

    #Time parameters
    tspan::Tuple = context[25]

    #To be passed as parameters to the iceflow! function
    p::Tuple = (dx, Γ, slope, slope_stag, width, width_stag, w₀, w₀_stag, λ, surface_h, bed_h, H_stag, grad_x_diff, flux_div, diffusivity, fs, ρ_g)

    #Callback functions for the mass balance and update the SIA_1D object 
    tstops, _ = define_callback_steps(tspan)
    stop_condition(u,t,integrator) = stop_condition_tstops(u,t,integrator, tstops) #closure

    #To be called during the callbacks
    function action!(integrator)
        get_mb!(mb, bed_h .+ integrator.u ,SIA_1D.yr,SIA_1D)
        integrator.u .+= (mb .* sec_in_month)
        integrator.u[integrator.u.<0.0] .= 0.0

        #writing the thickness solution every year as a property of the SIA_1D object
        setproperty!(SIA_1D, :t, integrator.t)
        setproperty!(SIA_1D.fls[0], :thick, np.array(integrator.u))

    end   

    #Defining the callback
    cb_MB = DiscreteCallback(stop_condition, action!)

    #Solving the problem         
    iceflow_prob = ODEProblem(iceflow!,H₀,tspan,tstops=tstops,p)
    iceflow_sol = solve(iceflow_prob,solver,callback=cb_MB, tstops=tstops, reltol=reltol,save_everystep=false,dense=false)

    return SIA_1D

end