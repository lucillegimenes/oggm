np=pyimport("numpy")


#Constants definition
const sec_in_day = 24.0 * 60.0 * 60.0
const sec_in_year = sec_in_day * 365.0
const sec_in_month = 2628000.0
const n = 3.0 #glen n
const ρ = 900.0
const g = 9.81

###############################################
###### SHALLOW ICE APPROXIMATION MODELS #######
###############################################


mutable struct SIA1Dmodel{F <: AbstractFloat, I <: Integer}
    nx::Union{I,Nothing}
    A::Union{F, Nothing}
    Γ::Union{F, Nothing}
    Γs::Union{F, Nothing}
    Δx::Union{I, Nothing}
    H::Union{Vector{F}, Nothing}
    H̅::Union{Vector{F}, Nothing}
    w::Union{Vector{F}, Nothing}
    w̅ ::Union{Vector{F}, Nothing}
    w0::Union{Vector{F}, Nothing}
    w0_stag::Union{Vector{F}, Nothing}
    λ::Union{Vector{F}, Nothing}
    S::Union{Vector{F}, Nothing}
    B::Union{Vector{F}, Nothing}
    dSdx::Union{Vector{F}, Nothing}
    ∇Sx::Union{Vector{F}, Nothing}
    D::Union{Vector{F}, Nothing}
    Fx::Union{Vector{F}, Nothing}
    Fxx::Union{Vector{F}, Nothing}
    mb::Union{Vector{F}, Nothing}
end 


function SIA1Dmodel(
    nx::Union{I,Nothing} = nothing,
    A::Union{F, Nothing} = nothing,
    Γ::Union{F, Nothing} = nothing,
    Γs::Union{F, Nothing} = nothing,
    Δx::Union{I, Nothing} = nothing,
    H::Union{Vector{F}, Nothing} = nothing, 
    H̅::Union{Vector{F}, Nothing} = nothing,
    w::Union{Vector{F}, Nothing} = nothing,
    w̅ ::Union{Vector{F}, Nothing} = nothing, 
    w0::Union{Vector{F}, Nothing} = nothing, 
    w0_stag::Union{Vector{F}, Nothing} = nothing, 
    λ::Union{Vector{F}, Nothing} = nothing,
    S::Union{Vector{F}, Nothing} = nothing, 
    B::Union{Vector{F}, Nothing} = nothing, 
    dSdx::Union{Vector{F}, Nothing} = nothing, 
    ∇Sx::Union{Vector{F}, Nothing} = nothing, 
    D::Union{Vector{F}, Nothing} = nothing, 
    Fx::Union{Vector{F}, Nothing}= nothing, 
    Fxx::Union{Vector{F}, Nothing}= nothing,
    mb::Union{Vector{F}, Nothing}= nothing) where {F <: AbstractFloat, I <: Integer}

    SIA1D_model = SIA1Dmodel{Float64,Int64}(nx, A, Γ, Γs, Δx, H, H̅, w, w̅, w0, w0_stag, λ, S, B, dSdx, ∇Sx, D, Fx, Fxx, mb)

    return SIA1D_model
end

function initialize_iceflow_model!(iceflow_model::SIA1Dmodel, SIA1D::Py)
    fl = SIA1D.fls[0]
    nx = pyconvert(Int64,fl.nx)
    iceflow_model.nx = pyconvert(Int64,fl.nx)
    iceflow_model.A = pyconvert(Float64,SIA1D.glen_a)
    iceflow_model.Γs = pyconvert(Float64,SIA1D.fs)*((ρ .* g).^n)
    iceflow_model.Γ = ((2.0 * iceflow_model.A * ((ρ .* g).^n) )/(n + 2.0))
    iceflow_model.Δx = pyconvert(Int64,fl.dx_meter)
    iceflow_model.H = pyconvert(Vector{Float64},fl.thick)
    iceflow_model.H̅ = zeros(Float64,nx-1)
    iceflow_model.w = pyconvert(Vector{Float64},fl.widths_m)
    iceflow_model.w̅ = zeros(Float64, nx-1)
    iceflow_model.w0 = pyconvert(Vector{Float64},fl._w0_m)
    iceflow_model.w0_stag = zeros(Float64, nx-1)
    avg!(iceflow_model.w0_stag, iceflow_model.w0)
    iceflow_model.λ = pyconvert(Vector{Float64},fl._lambdas)
    iceflow_model.S = pyconvert(Vector{Float64},fl.surface_h)
    iceflow_model.B = pyconvert(Vector{Float64},fl.bed_h)
    iceflow_model.dSdx = zeros(Float64, nx-1)
    iceflow_model.∇Sx = zeros(Float64, nx + 1)
    iceflow_model.D = zeros(Float64, nx + 1)
    iceflow_model.Fx = zeros(Float64,nx + 1)
    iceflow_model.Fxx = zeros(Float64,nx)
    iceflow_model.mb = zeros(Float64,nx)
end

###############################################
######    VARIABLE STORAGE FOR OGGM     #######
###############################################

mutable struct diag
    volume_m3::Vector{Union{Float64,Py}}
    area_m2::Vector{Union{Float64,Py}}
    length_m::Vector{Union{Float64,Py}}
    calving_m3::Vector{Union{Float64,Py}}
    calving_rate_myr::Vector{Union{Float64,Py}}
    volume_bsl_m3::Vector{Union{Float64,Py}}
    volume_bwl_m3::Vector{Union{Float64,Py}}
    area_m2_min_h::Vector{Union{Float64,Py}}
end

mutable struct geom
    sects::Vector{Matrix{Union{Float64,Py}}}
    widths::Vector{Matrix{Union{Float64,Py}}}
    buckets::Vector{Vector{Union{Float64,Py}}}
end 

mutable struct fl_diag
    thick_fl::Vector{Matrix{Union{Float64,Py}}} 
    volume_bsl_fl::Vector{Vector{Union{Float64,Py}}} 
    volume_bwl_fl::Vector{Vector{Union{Float64,Py}}}
end 

function diag(nₜm::Int64,fac::Int64,diff::Int64, SIA1D::Py,l_var::PyList{Any},dyn_spin_thick::Float64)
    v = Vector{Float64}(zeros(nₜm))
    diag_jl = diag(v,v,v,v,v,v,v,v)
    if diff >= 0               
        #Diagnostic at year 0 (and more if there is a spinup) 
        for i in 1:(diff*fac)+1
            update_diag!(diag_jl,SIA1D,l_var,dyn_spin_thick,i)
        end 
    end

    return diag_jl
end  

function update_diag!(diag_jl::diag,SIA1D::Py,l_var::PyList{Any},dyn_spin_thick::Float64,j::Int)
    diag_jl.volume_m3[j] = SIA1D.volume_m3
    diag_jl.area_m2[j] = SIA1D.area_m2
    diag_jl.length_m[j] = SIA1D.length_m
    diag_jl.calving_m3[j] = SIA1D.calving_m3_since_y0
    diag_jl.calving_rate_myr[j] = SIA1D.calving_rate_myr
    diag_jl.volume_bsl_m3[j] = SIA1D.volume_bsl_m3
    diag_jl.volume_bwl_m3[j] = SIA1D.volume_bwl_m3
    if "area_min_h" in l_var
        diag_jl.area_m2_min_h[j] = sum([sum(fl.bin_area_m2[fl.thick > dyn_spin_thick]) for fl in SIA1D.fls])
    end 
end 

function geom_fl(do_geom::Bool,do_fl_diag::Bool,nₜ::Int64,nx::Int64,SIA1D::Py,diff::Int64,is_tidewater::Bool)
    m = [zeros(Float64,nₜ,nx) for fl in SIA1D.fls]
    v =[zeros(Float64,nₜ) for fl in SIA1D.fls]
    geom_var = geom(m,m,v)
    fl_var = fl_var= fl_diag(m,v,v)

    #Storage for geometry
    if do_geom || do_fl_diag
        if diff >= 0               
            for i in 1:diff+1
                update_geom!(geom_var,SIA1D,is_tidewater,i)
            end 
        end 

        if do_fl_diag
            if diff >= 0               
                for i in 1:diff+1
                    update_fl_diag!(fl_var,SIA1D,i)
                end 
            end     

        end 
    end 

    return geom_var, fl_var
end 

function update_geom!(geom_var::geom,SIA1D::Py,is_tidewater::Bool,j::Int)
    for (s,w,b,fl) in zip(geom_var.sects,geom_var.widths,geom_var.buckets,SIA1D.fls)
        @views s[j,:] .= fl.section
        @views w[j,:] .= fl.widths_m
        if is_tidewater
            try 
                b[j] = fl.calving_bucket_m3
            catch 
                println("AttributeError")
            end 
        end
    end 
end 

function update_fl_diag!(fl_var::fl_diag,SIA1D::Py,j::Int)
    for (t,vs,vw,fl) in zip(fl_var.thick_fl,fl_var.volume_bsl_fl,fl_var.volume_bwl_fl,SIA1D.fls)
        @views t[j,:] .= fl.thick
        vs[j] = fl.volume_bsl_m3
        vw[j] = fl.volume_bwl_m3
    end 
end 

###############################################
######    SIMULATION PARAMETERS         #######
###############################################

mutable struct SimulationParameters{F <: AbstractFloat, I <: Integer} 
    y₀::Union{I,Nothing}
    y₀slf::Union{I,Nothing}
    y₁::Union{I,Nothing}
    yr::Union{F,Nothing}
    nₜ::Union{I, Nothing}
    diff::Union{I, Nothing}
    tspan::Union{Tuple, Nothing}
    mb_step::Union{String, Nothing}
    nₜm::Union{I, Nothing}
    fact_m::Union{I, Nothing}
end

function SimulationParameters(
    y₀::Union{I,Nothing} = nothing, 
    y₀slf::Union{I,Nothing} = nothing, 
    y₁::Union{I,Nothing} = nothing, 
    yr::Union{F,Nothing} = nothing, 
    nₜ::Union{I, Nothing} = nothing,
    diff::Union{I, Nothing} = nothing,
    tspan::Union{Tuple, Nothing} = nothing,
    mb_step::Union{String, Nothing} = nothing,
    nₜm::Union{I, Nothing} = nothing,
    fact_m::Union{I, Nothing} = nothing) where {F <: AbstractFloat, I <: Integer}

    Simulation_Parameters = SimulationParameters{Float64,Int64}(y₀, y₀slf, y₁, yr, nₜ, diff, tspan, mb_step, nₜm, fact_m)
                                                    
    return Simulation_Parameters
end

function initialize_sim_params!(Simulation_Parameters::SimulationParameters, SIA1D::Py, y0::Int64, y1::Int64, mb_step::String)
    Simulation_Parameters.y₀ = pyconvert(Int64,y0)
    Simulation_Parameters.y₀slf = pyconvert(Int64,SIA1D.y0)
    Simulation_Parameters.y₁ = pyconvert(Int64,y1)
    Simulation_Parameters.yr = Simulation_Parameters.y₀slf
    Simulation_Parameters.nₜ = Int(y1-y0) + 1
    Simulation_Parameters.diff = Simulation_Parameters.y₀slf - Simulation_Parameters.y₀
    Simulation_Parameters.tspan = (0.0 , (Simulation_Parameters.y₁ - Simulation_Parameters.y₀slf)*sec_in_year)
    Simulation_Parameters.mb_step = mb_step
    if Simulation_Parameters.mb_step =="annual"
        Simulation_Parameters.nₜm = Simulation_Parameters.nₜ
        Simulation_Parameters.fact_m = 1
    else 
        Simulation_Parameters.nₜm = (Simulation_Parameters.nₜ-1)*12 +1
        Simulation_Parameters.fact_m = 12
    end 

end 


