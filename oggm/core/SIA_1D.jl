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

function glacier_evolution_store(;
                            SIA_1D::Py, 
                            solver, 
                            reltol::Float64,
                            y0::Int64, 
                            y1::Int64, 
                            l_var::PyList{Any}, 
                            do_geom::Int64, 
                            do_fl_diag::Int64,
                            mb_step::String,
                            dyn_spin_thick)
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
    nx = context[1]
    dx = context[2]

    #Thickness
    H₀ = context[3]
    H_stag = context[4]
    
    #Working with trapezoidal bed shape 
    width = context[5]
    width_stag = context[6]
    w₀_stag = context[7]
    w₀ = context[29]
    λ = context[30]

    bed_h = context[8]

    #Pre-allocation
    surface_h = context[9]
    surface_exp = context[10]
    slope = context[11]
    slope_stag = context[12]
    grad_x_diff = context[13]
    flux_div = context[14]
    diffusivity = context[15]
    mb = context[16]

    #Parameters for later computation
    A = context[17]
    ρ_g = context[18]
    Γ = context[19]
    fs= context[20]

    #Time parameters
    y₀ = context[21]
    y₀slf= context[22]
    y₁ = context[23]

    tspan = context[24]
    nₜ = context[25]
    nₜm = context[26]

    diff = context[27]
    fac = context[28]

    #To be passed as parameters to the iceflow! function
    p = (dx, Γ, slope, slope_stag, width, width_stag, w₀_stag, surface_h, surface_exp, bed_h, H_stag, grad_x_diff, flux_div, diffusivity,
        SIA_1D, fs, ρ_g, w₀, λ)

    ## Diagnostic         
    #Storage for diagnostic variables 
    diag_jl = Dict("volume_m3" => Vector{Float64}(zeros(nₜm)),
                   "area_m2" => Vector{Float64}(zeros(nₜm)),
                   "length_m"=> Vector{Float64}(zeros(nₜm)),
                   "calving_m3"=> Vector{Float64}(zeros(nₜm)),
                   "calving_rate_myr"=> Vector{Float64}(zeros(nₜm)),
                   "volume_bsl_m3"=> Vector{Float64}(zeros(nₜm)),
                   "volume_bwl_m3"=> Vector{Float64}(zeros(nₜm)),
                   "area_m2_min_h"=> Vector{Float64}(zeros(nₜm)))
    if diff >= 0               
        #Diagnostic at year 0 (and more if there is a spinup) 
        for i in 1:(diff*fac)+1
            update_diag!(diag_jl,SIA_1D,l_var,dyn_spin_thick,i)
        end 
    end 

    ##Geometry and flowline diagnostic
    do_geom = Bool(do_geom)
    do_fl_diag = Bool(do_fl_diag)
    sects = widths = buckets = thick_fl = volume_bsl_fl = volume_bwl_fl = nothing
    #Storage for geometry
    if do_geom || do_fl_diag
        sects = [zeros(Float64,nₜ,nx) for fl in SIA_1D.fls]
        widths = [zeros(Float64,nₜ,nx) for fl in SIA_1D.fls]
        buckets = [zeros(Float64,nₜ) for fl in SIA_1D.fls]
        if diff >= 0               
            for i in 1:diff+1
                update_geom!(sects,widths,buckets,SIA_1D,i)
            end 
        end 

        if do_fl_diag
            thick_fl = [zeros(Float64,nₜ,nx) for fl in SIA_1D.fls]
            volume_bsl_fl = [zeros(Float64,nₜ) for fl in SIA_1D.fls]
            volume_bwl_fl = [zeros(Float64,nₜ) for fl in SIA_1D.fls]
            if diff >= 0               
                for i in 1:diff+1
                    update_fl_diag!(thick_fl,volume_bsl_fl,volume_bwl_fl,SIA_1D,i)
                end 
            end     

        end 
    end 


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
                        update_geom!(sects,widths,buckets,SIA_1D,tᵢ)
                        if do_fl_diag
                            update_fl_diag!(thick_fl,volume_bsl_fl,volume_bwl_fl,SIA_1D,tᵢ)
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
                        update_geom!(sects,widths,buckets,SIA_1D,tᵢ)
                        if do_fl_diag
                            update_fl_diag!(thick_fl,volume_bsl_fl,volume_bwl_fl,SIA_1D,tᵢ)
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

    return SIA_1D,diag_jl, sects, widths, buckets, thick_fl, volume_bsl_fl, volume_bwl_fl
    
end


#### FUNCTIONS FOR OGGM OUTPUTS ####

function update_diag!(diag_jl::Dict{String, Vector{Float64}},SIA_1D::Py,l_var::PyList{Any},dyn_spin_thick,j::Int)
    diag_jl["volume_m3"][j] = pyconvert(Float64,SIA_1D.volume_m3)
    diag_jl["area_m2"][j] = pyconvert(Float64,SIA_1D.area_m2)
    diag_jl["length_m"][j] = pyconvert(Float64,SIA_1D.length_m)
    diag_jl["calving_m3"][j] = pyconvert(Float64,SIA_1D.calving_m3_since_y0)
    diag_jl["calving_rate_myr"][j] = pyconvert(Float64,SIA_1D.calving_rate_myr)
    diag_jl["volume_bsl_m3"][j] = pyconvert(Float64,SIA_1D.volume_bsl_m3)
    diag_jl["volume_bwl_m3"][j] = pyconvert(Float64,SIA_1D.volume_bwl_m3)
    if "area_min_h" in l_var
        diag_jl["area_m2_min_h"][j] = pyconvert(Float64,sum([sum(fl.bin_area_m2[fl.thick > dyn_spin_thick]) for fl in SIA_1D.fls]))
    end 
end 


function update_geom!(sects::Vector{Matrix{Float64}},widths::Vector{Matrix{Float64}},buckets::Vector{Vector{Float64}},SIA_1D::Py,j::Int)
    for (s,w,b,fl) in zip(sects,widths,buckets,SIA_1D.fls)
        s[j,:] .= pyconvert(Vector{Float64},fl.section)
        w[j,:] .= pyconvert(Vector{Float64},fl.widths_m)
        if pyconvert(Bool,SIA_1D.is_tidewater)
            try 
                b[j] = pyconvert(Float64,fl.calving_bucket_m3)
            catch 
                println("AttributeError")
            end 
        end
    end 
end 

function update_fl_diag!(thick_fl::Vector{Matrix{Float64}},volume_bsl_fl::Vector{Vector{Float64}},volume_bwl_fl::Vector{Vector{Float64}},SIA_1D::Py,j::Int)
    for (t,vs,vw,fl) in zip(thick_fl,volume_bsl_fl,volume_bwl_fl,SIA_1D.fls)
        t[j,:] .= pyconvert(Vector{Float64},fl.thick)
        vs[j] = pyconvert(Float64,fl.volume_bsl_m3)
        vw[j] = pyconvert(Float64,fl.volume_bwl_m3)
    end 
end 

#### ####

#### FONCTIONS FOR THE CALLBACKS ####

function get_mb!(mb::Vector{Float64}, heights::Vector{Float64},y::Py,SIA_1D::Py)
    mb .= pyconvert(Vector{Float64},SIA_1D.get_mb(heights,y,fl_id=0))     
end

function define_callback_steps(tspan)
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

    dx = pyconvert(Int64,fl.dx_meter)
    nx = pyconvert(Int64,fl.nx)
    λ = pyconvert(Vector{Float64},fl._lambdas)
    
    #Thickness
    H₀ = pyconvert(Vector{Float64},fl.thick)
    H_stag = zeros(Float64,nx-1)
    
    #Working with trapezoidal bed shape 
    width = zeros(Float64,nx)
    width_stag = zeros(Float64,nx-1)
    w₀ = pyconvert(Vector{Float64},fl._w0_m)
    w₀_stag = zeros(Float64,nx-1)
    avg!(w₀_stag,w₀)

    bed_h = pyconvert(Vector{Float64},fl.bed_h)

    #Pre-allocation
    surface_h = zeros(Float64,nx)
    surface_exp = zeros(Float64,nx+2)
    slope = zeros(Float64,nx-1)
    slope_stag = zeros(Float64,nx+1)
    grad_x_diff = zeros(Float64,nx+1)
    flux_div = zeros(Float64,nx)
    diffusivity = zeros(Float64,nx+1)
    mb = zeros(Float64,nx)

    #Parameters for later computation
    A = pyconvert(Float64,SIA_1D.glen_a)
    ρ_g = (ρ .* g).^glen_n
    Γ = ((2.0 * A * ρ_g )/(glen_n + 2.0))
    fs= pyconvert(Float64,SIA_1D.fs)

    #Time parameters
    y₀ = pyconvert(Int64,y0) #Starting year of the diagnostic variable datasets
    y₀slf= pyconvert(Int64,SIA_1D.y0) #Starting year for the actual simulation
    y₁ = pyconvert(Int64,y1)

    tspan = (0.0 , (y₁ - y₀slf)*sec_in_year)
    nₜ = Int(y1-y0) + 1 #Number of years for the output

    diff = y₀slf-y₀ #This parameter is useful in case of spinup (i.e when y₀slf =/= y₀)

    #Required MB step
    if mb_step =="annual"
        nₜm = nₜ
        fac = 1
    else 
        nₜm = (nₜ-1)*12 +1
        fac = 12
    end 

    context = (nx, dx, H₀, H_stag, width, width_stag, w₀_stag, bed_h, surface_h, surface_exp, slope, slope_stag, grad_x_diff, flux_div, diffusivity, mb,
                A, ρ_g, Γ, fs, y₀, y₀slf, y₁, tspan, nₜ, nₜm, diff, fac, w₀, λ)
    
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
    w₀_stag::Vector{Float64},
    surface_h::Vector{Float64},
    surface_exp::Vector{Float64},
    bed_h::Vector{Float64},
    H_stag::Vector{Float64}, 
    grad_x_diff::Vector{Float64}, 
    flux_div::Vector{Float64},
    diffusivity::Vector{Float64},
    SIA_1D::Py,
    fs::Float64,
    ρ_g::Float64,
    w₀::Vector{Float64},
    λ:: Vector{Float64}= p

    #read out variable from current flowline 

    #setproperty!(SIA_1D.fls[0], :thick, np.array(H))
    #width .= pyconvert(Vector{Float64},SIA_1D.fls[0].widths_m)
    width .= w₀ .+ (λ.* H)

    surface_h .= bed_h .+ H

    #Staggered thickness
    avg!(H_stag,H)

    #Staggered width
    avg!(width_stag,width)    

    #(Staggered) slope
    diff1!(slope,surface_h,dx)

    # Diffusivity
    @views @. diffusivity[2:end-1] = ((slope^2.0)^((glen_n-1)/2.0)) * ((width_stag + w₀_stag)/2.0) * (Γ * H_stag^(glen_n.+2) + fs*ρ_g*(H_stag^glen_n))

    surface_exp .=vcat(surface_h[1],surface_h,surface_h[end])
    diff1!(slope_stag,surface_exp,dx)

    grad_x_diff .= .-diffusivity .* slope_stag
    diff1!(flux_div,grad_x_diff,dx)

    dH .=  .-flux_div./width 

    # Clip negative ice thickness values
    @views H[H.<0.0] .= 0.0
    @assert H[end] .< 10.0 "Glacier exceeding boundaries! at time $(t/sec_in_year)"

end

#Basically like glacier_evolution_store but without storing anything
function glacier_evolution(;
    SIA_1D::Py, #IceflowJuliaModel object
    solver, #Solver to use
    reltol::Float64, #Relative tolerance
    y0::Int64, #starting year of the simulation
    y1::Int64) #ending year of the simulation

    # read out variables from current flowline 
    fl = SIA_1D.fls[0]
    dx = pyconvert(Int64,fl.dx_meter)
    nx = pyconvert(Int64,fl.nx)

    H₀ = pyconvert(Vector{Float64},fl.thick)
    H_stag = zeros(Float64,nx-1)

    width = zeros(Float64,nx)  #Assuming rectangular or trapezoidal bed shape for now
    width_stag = zeros(Float64,nx-1)

    w₀ = pyconvert(Vector{Float64},fl._w0_m)
    w₀_stag = zeros(Float64,nx-1)
    avg!(w₀_stag,w₀)
    bed_h = pyconvert(Vector{Float64},fl.bed_h)

    #Pre-allocation
    surface_h = zeros(Float64,nx)
    surface_exp = zeros(Float64,nx+2)
    slope = zeros(Float64,nx-1)
    slope_stag = zeros(Float64,nx+1)

    grad_x_diff = zeros(Float64,nx+1)
    flux_div = zeros(Float64,nx)

    diffusivity = zeros(Float64,nx+1)

    mb = zeros(Float64,nx)

    #Parameters for computation
    A = pyconvert(Float64,SIA_1D.glen_a)
    ρ_g = (ρ .* g).^glen_n
    Γ = ((2.0 * A * ρ_g )/(glen_n + 2.0))
    fs= pyconvert(Float64,SIA_1D.fs)

    #Time
    y₀ = pyconvert(Int64,y0)
    y₁ = pyconvert(Int64,y1)

    tspan = (0.0 , (y₁ - y₀)*sec_in_year)


    #To be passed as parameters to the iceflow! function
    p = (dx, Γ, slope, slope_stag, width, width_stag, w₀_stag, surface_h, surface_exp, bed_h, H_stag, grad_x_diff, flux_div, diffusivity,
    SIA_1D, fs, ρ_g)


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