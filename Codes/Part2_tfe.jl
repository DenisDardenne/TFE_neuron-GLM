using Plots
println("Start loading lib_fct.jl")


# Function to create a custom x-axis
function create_custom_x_axis(x, skip_start, skip_end)
    x_new = []
    for xi in x
        if xi < skip_start
            push!(x_new, xi)
        elseif xi > skip_end
            push!(x_new, xi - (skip_end - skip_start))
        end
    end
    return x_new
end



using DifferentialEquations, LaTeXStrings, DelimitedFiles  # Polynomials, GraphPlot, ColorSchemes
using Statistics, Random, Printf, LinearAlgebra # StatsPlots, ProgressMeter, SimpleWeightedGraphs, Compose

using Clustering, ColorSchemes

neuron_type = ["STG","DA"]
nt = neuron_type[2]
DIC = true

if nt=="STG"
    FakeDoublets = "doublets"
else
    FakeDoublets = "triplets"
end

include("CodeArthurFyon/P3/CORR_2024-main/$(nt)/FYON_2022_$(nt)_models.jl") # Include STG model gating functions
# The model are available in his Github page

default(fmt = :png)    
include("lib_fct.jl");
println("End loading lib_fct.jl")






g_all_spiking = readdlm("CodeArthurFyon/P3/CORR_2024-main/$(nt)/data/g_all_spiking.dat")
    # g_all_spiking_DIC = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/g_all_spiking_DIC.dat")
    # g_all_spiking_leak = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/g_all_spiking_leak.dat")

g_all_doublets = readdlm("CodeArthurFyon/P3/CORR_2024-main/$(nt)/data/g_all_$(FakeDoublets).dat")
    # g_all_doublets_DIC = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/g_all_doublets_DIC.dat")
    # g_all_doublets_leak = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/g_all_doublets_leak.dat")

g_all_bursting = readdlm("CodeArthurFyon/P3/CORR_2024-main/$(nt)/data/g_all_bursting.dat")
    # g_all_bursting_DIC = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/g_all_bursting_DIC.dat")
    # g_all_bursting_leak = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/g_all_bursting_leak.dat")
burstiness_doublets = readdlm("CodeArthurFyon/P3/CORR_2024-main/$(nt)/data/burstiness_$(FakeDoublets).dat")
    # burstiness_doublets_plot = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/burstiness_doublets_plot.dat")
    # burstiness_doublets_DIC = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/burstiness_doublets_DIC.dat")
    # burstiness_doublets_DIC_plot = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/burstiness_doublets_DIC_plot.dat")
    # burstiness_doublets_leak = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/burstiness_doublets_leak.dat")
    # burstiness_doublets_leak_plot = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/burstiness_doublets_leak_plot.dat")
burstiness_bursting = readdlm("CodeArthurFyon/P3/CORR_2024-main/$(nt)/data/burstiness_bursting.dat")
    # burstiness_bursting_DIC = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/burstiness_bursting_DIC.dat")
    # burstiness_bursting_leak = readdlm("CodeArthurFyon/P3/CORR_2024-main/STG/data/burstiness_bursting_leak.dat");
#
# Loop over all neurons in the set
function SimulateODE(G, I::Function, tspan, dt, model )
  
    # Parameter vector for simulations
    if model == "STG"
        (gNa, gCaT, gCaS, gA, gKCa, gKd, gH, gleak) = G[:]
        p = [I, gNa, gCaT, gCaS, gA, gKCa, gKd, gH, gleak, C]
        x0  = [V0, mNa_inf(V0), hNa_inf(V0), mCaT_inf(V0), hCaT_inf(V0), mCaS_inf(V0), 
              hCaS_inf(V0), mA_inf(V0), hA_inf(V0), mKCa_inf(V0, Ca0), mKd_inf(V0), mH_inf(V0), Ca0]
        prob = ODEProblem(STG_ODE_HS, x0, tspan, p)
    elseif model == "DA"
        (gNa, gKd, gCaL, gCaN, gERG, gNMDA, gleak) = G[:]
        p = [I, gNa, gKd, gCaL, gCaN, gERG, gNMDA, gleak, C]
        x0 = [V0, m_inf(V0), h_inf(V0), n_inf(V0), mCaL_inf(V0), mCaN_inf(V0), 0., 0.]
        prob = ODEProblem(DA_ODE_HS, x0, tspan, p)
    else
        println("Error in SimulateODE")
        return 0
    end

    sol = solve(prob, verbose=false)
  
    # Removing transient part
    t = tspan[1]:dt:tspan[2]
    S = sol(t)
    V = S[1,:]

    return t, V
end

function makeSim(Gmatrix, name,nt)
    n_it = size(Gmatrix,1)
    if nt == "STA"
        tspan = (0,5000)
        skip = 5000
    elseif nt == "DA"
        tspan = (0,20000)
        skip = 7500
    else
        println("Error in makeSim ")
        return 0
    end
        # I(t) = (t > 1000 && t < 4000) ? 0 : -10
    I(t) = 0
    v_th = 30
    dt = 0.1
    global step = 1
    lk1 = ReentrantLock()

    Threads.@threads for i in 1:1:n_it
        lock(lk1) do
            println(name, " ", step, " / ", n_it)
            global step = step + 1
        end
        println(name, " : ", i, " / ", n_it)
        ~, V = SimulateODE(Gmatrix[i,:], I, tspan, dt, nt)
        spike = convert.(Int, V .>= v_th)
        for id in length(spike):-1:2
            if spike[id] == 1 && spike[id-1] == 1
                spike[id] = 0
            end 
        end
        save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Spikesof$(name)"],"spike_$(i).csv",spike[skip:end])
        if i == 1
            save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Iapp"],"Iapp.csv",I.(dt:dt:tspan[2])[skip:end])
            save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","N_SIM"],"$(name).csv",n_it)
        end
    end
end

function make_draft(name,nt)
    n_it = read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/N_SIM/$(name).csv")[1]
    step = 50
    dt = 0.1
    draft = plot()
    siz = 0.35
    for i in convert.(Int,1:step:n_it)
        println(name, " : ", round(i/step), " / ", round(n_it/step))
        spike = read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/Spikesof$(name)/spike_$(i).csv",1)
        Ns = sum(spike)
        for j in 1:length(spike)
            if spike[j] == 1
                X = round(j*dt)
                Y = round(i/step)
                plot!([X,X],[Y-siz,Y+siz],label="", color="black")
            end
        end
    end
    folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
    savefig(draft, joinpath(folder_path, "QuickDraft_$(name).svg"))
end

function give_nbasis(Name, NAME)
    if Name == NAME[1]
        nbasK = 3  # 5
        nbasH = 7  # 4
    elseif Name == NAME[2]
        nbasK = 3  # 7
        nbasH = 7  # 6
    else # bursting
        nbasK = 3  # 9
        nbasH = 7  # 7
    end
    return nbasK, nbasH
end

function obtain_H(path_folder, nieme, basis_H=0, THE_BASE=0)
    if THE_BASE == 0
        if basis_H in [0,1]
        hbasis = read_data(joinpath(path_folder, "hbasis.csv"))
        else
            hbasis = basis_H
        end
    else
        hbasis = THE_BASE
    end
    prs = read_data(joinpath(path_folder, "prs$(nieme).csv"))
    prsH = prs[end-size(hbasis,2):end-1]
    h = hbasis * prsH
    dc = prs[end]
    if basis_H == 0
        return h[1:convert(Int,round(length(h)/2))], dc
    elseif basis_H == 1
        return h[1:convert(Int,round(length(h)/2))], hbasis
    else
        if nt == "DA"
            return h[1:convert(Int,round(length(h)/2))]
        else
            return h[:]
        end
    end
end

function rescale_to_01(values)
    min_val = minimum(values)
    max_val = maximum(values)
    scaled_values = (values .- min_val) ./ (max_val - min_val)
    return scaled_values
end

function built_bad_class(C1, numC1, name_to_save, Name_vect)

    folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Classification_Fig"], "", "", 1)
    j = 1
    part = 1
    Bad_index = []
    PLT_miss = plot(size=(400,400))
    for i in 1:length(C1)
        if C1[i] != numC1 # plotter les erreurs
            push!(Bad_index, i)
            ibis = i
            while ibis > 500
                ibis = ibis-500
            end
            spike = read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/Spikesof$(Name_vect[i])/spike_$(ibis).csv", 1)[1:end-1]
            for x in 1:length(spike)
                if spike[x] == 1
                    plot!([x,x].*dt, [j-0.35, j+0.35], label="", color="black")
                end
            end
            j = j + 1
            if j == 21
                plot!(xlabel=L"Time~[ms]")
                savefig(PLT_miss, joinpath(folder_path, "part$(part)_$(name_to_save)"))
                PLT_miss = plot(size=(400,400))
                j = 1
                part = part + 1
            end
        end
    end
    plot!(xlabel=L"Time~[ms]")
    savefig(PLT_miss, joinpath(folder_path, "part$(part)_$(name_to_save)"))

    return Bad_index
end

if nt == "STG"
     C     = 1. # Membrane capacitance
     VNa   = 50 # Sodium reversal potential
     VK    = -80 # Potassium reversal potential
     VCa   = 80 # Calcium reversal potential
     VH    = -20 # HCN channels reversal potential
     Vleak = -50 # Leak reversal potential
     V0  = -70.
     Ca0 = 0.5
elseif nt == "DA"
    # Definition of reversal potential values (in mV), [Mg] and membrane capacitance
     VNa   = 60. # Sodium reversal potential
     VK    = -85. # Potassium reversal potential
     VCa   = 60. # Calcium reversal potential
     VNMDA = 0. # NMDA reversal potential
     Vleak = -50. # Reversal potential of leak channels
     Mg    = 1.4 # Mg concentration
     C     = 1. # Membrane capacitance
     V0 = -90.
else
    println("Error in the constant definition")
end

# Definition of voltage range for the DICs
 Vmin = -60 
 Vmax = 0



global lk_access_and_create_folder = ReentrantLock()


generate_pattern = false
quickplot = false
fitGLM = false
analyse_filter = false
classification_last = false
same_b_diff_h = false


NAME = ["spiking","doublets","bursting"]

dt= 0.1


if generate_pattern
    # Generate pattern with ODE
    makeSim(g_all_spiking, NAME[1],nt)
    makeSim(g_all_doublets, NAME[2],nt)
    makeSim(g_all_bursting, NAME[3],nt)
end

if quickplot
    for n in NAME
        make_draft(n,nt)
    end
end

if fitGLM 
    for Name in NAME
        nbasK, nbasH = give_nbasis(Name, NAME)
        nkt = 100; # number of ms in stim filter
        kbasprs = Dict(
            :neye => 0,  # number of "identity" basis vectors near time of spike
            :ncos => nbasK,  # number of raised-cosine vectors to use
            :kpeaks => [0.1, round(nkt / 1.2)],  # position of first and last bump relative to identity bumps
            :b => 10  # how nonlinear to make spacings (larger values make it more linear)
                )
        if nt == "STG"
            time_step = 150
        elseif nt == "DA"
            time_step = 1000
        else
            println("Error for time_step")
            return 0
        end
        ihbasprs = Dict(
            :ncols => nbasH,  # number of basis vectors for post-spike kernel
            :hpeaks => [0.1, time_step],  # peak location for the first and last vectors, in ms
            :b => 10,  # how nonlinear to make spacings (larger values make it more linear)
            :absref => 1  # absolute refractory period, in ms
        )
        maxIter =  100  #  1000;  # max number of iterations for fitting, also used for maximum number of function evaluations(MaxFunEvals)
        tolFun = 1e-12;  # function tolerance for fitting

        Iapp = read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/Iapp/Iapp.csv")
        solver = [NewtonTrustRegion()]
        L = length(solver)
        N = (read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/N_SIM/$(Name).csv")[1]) 
        global step = 1
        lk1 = ReentrantLock()

        save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","SaveInformationFitting","Basis_info"],"nbasK.csv",nbasK)
        save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","SaveInformationFitting","Basis_info"],"nkt.csv",nkt)
        save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","SaveInformationFitting","Basis_info"],"nbasH.csv",nbasH)
        save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","SaveInformationFitting","Basis_info"],"time_step.csv",time_step)


        Threads.@threads for i in convert.(Int,1:1:N)
            lock(lk1) do
                println(Name, " : ", step, " / ", N)
                global step = step + 1
            end
            spike = read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/Spikesof$(Name)/spike_$(i).csv",1)[1:end-1]
            k, h, dc, prs, kbasis, hbasis = [],[],[],[],[],[]
            Solv_value, Solv_Gnorm, Solv_metaData = [],[],[]
            for ifor in 1:L
                k_i, h_i, dc_i, prs_i, kbasis_i, hbasis_i, trace_i = fit_glm(Iapp,spike,dt,nkt,kbasprs,ihbasprs,[],maxIter,tolFun, solver[ifor], "", "LLfit");
                push!(k, k_i), push!(h, h_i), push!(dc, dc_i), push!(prs, prs_i), push!(kbasis, kbasis_i), push!(hbasis, hbasis_i)
                for j in 1:length(trace_i)
                    push!(Solv_value, trace_i[j].value), push!(Solv_Gnorm, trace_i[j].g_norm), push!(Solv_metaData, trace_i[j].metadata["time"])
                end
            end
            k, h, dc, prs, kbasis, hbasis  = k[1], h[1], dc[1], prs[1], kbasis[1], hbasis[1];
            lock(lk1) do
                save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","SaveInformationFitting","$(Name)"],"prs$(i).csv",prs)
                if i == 1
                    save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","SaveInformationFitting","$(Name)"],"kbasis.csv",kbasis)
                    save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","SaveInformationFitting","$(Name)"],"hbasis.csv",hbasis)
                end
                save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","SaveInformationFitting","$(Name)"],"Solv_value$(i).csv",Solv_value)
            end
        end
    end
end


H_carac = zeros(1500,21)
if analyse_filter

    if false && isdir(joinpath("SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"))
        rm(joinpath("SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"), recursive=true)
    end
    DCs = []
    for Name in NAME
        println(Name)
        path_folder = "SavingOfComputation/WithArthurFyonCodes_$(nt)/SaveInformationFitting/$(Name)"
        h, hbasis = obtain_H(path_folder, 1, 1)

        N = convert(Int,(read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/N_SIM/$(Name).csv")[1]) )
        if nt == neuron_type[1]
            sampling = 1:1:length(h)
            sampling_for_plot = 1:10:length(h)
        else
            sampling = 1:1:length(h)
            sampling_for_plot = 1:100:length(h)
        end
        xh = sampling_for_plot.*dt

        H = zeros(N+1, length(h))
        DC = zeros(N+1, 1)
        prs_SCORE = []

        for i in convert.(Int,1:N)
            println("Loading of filter ", i, " over ", N, " of ", Name)
            h, dc = obtain_H(path_folder, i, 0, hbasis)
            score = read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/SaveInformationFitting/$(Name)/Solv_value$(i).csv")
            H[i,:] = h[sampling]
            DC[i] = dc
            push!(prs_SCORE,score)
        end

        H[end,:] =median(H[1:end-1,:],dims=1)
        DC[end] = median(DC[1:end-1])
        push!(DCs,DC[1:end-1])

        save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","SaveInformationFitting","$(Name)_median"],"H.csv",H[end,:])
        save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","SaveInformationFitting","$(Name)_median"],"DC.csv",DC[:])

        if Name == "spiking"
            burstiness = zeros(N,1)
            for i in 1:N
                burstiness[i] = length(read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/Spikesof$(Name)/spike_$(i).csv")[1:end-1])
            end
        elseif Name == "doublets"
            burstiness = readdlm("CodeArthurFyon/P3/CORR_2024-main/$(nt)/data/burstiness_$(FakeDoublets).dat")
        elseif Name == "bursting"
            burstiness = readdlm("CodeArthurFyon/P3/CORR_2024-main/$(nt)/data/burstiness_bursting.dat")
        end

        if true
            to_plot = randperm(length(burstiness))[1:20]

            mySalmon = RGB{Float64}(243. / 255., 124. / 255., 130. / 255.)
            myYellow  = RGB{Float64}(253. / 255., 211. / 255., 44. / 255.)
            mycmap = ColorScheme([mySalmon, myYellow]);
            my_col = mySalmon-myYellow
            diff_col = rescale_to_01(burstiness[to_plot[:]]) .* my_col; 
            COL = diff_col .+ myYellow;

            # Classic
            Hred = H[:,sampling_for_plot]
            P_full = plot(size=(400,400))
            for i in 1:length(to_plot)
                plot!(reverse(-xh), reverse(Hred[to_plot[i],:]'[:]), label="", size=(400,400),color=COL[i])
            end
            plot!(reverse(-xh),reverse(Hred[end,:]),label=L"Median~filter",color="black", linestyle=:dot, xlabel=L"Time~[ms]")
            plot!(xtickfontsize=12, ytickfontsize=12,legendfontsize=12,xlabelfontsize=16, ylabelfontsize=16)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(P_full, joinpath(folder_path, "Comp_Filter$(Name).pdf"))

            # Normed
            Hred = H./(maximum(H,dims=2).-minimum(H,dims=2))
            Hred = Hred[:,sampling_for_plot]
            P_full = plot(size=(400,400))
            for i in 1:length(to_plot)
                plot!(reverse(-xh), reverse(Hred[to_plot[i],:]'[:]), label="", size=(400,400),color=COL[i])
            end
            plot!(reverse(-xh),reverse(Hred[end,:]),label=L"Median~filter",color="black", linestyle=:dot, xlabel=L"Time~[ms]")
            plot!(xtickfontsize=12, ytickfontsize=12,legendfontsize=12,xlabelfontsize=16, ylabelfontsize=16)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(P_full, joinpath(folder_path, "Comp_Filter$(Name)_reduced.pdf"))

            # TanH
            Hred = tanh.(H)
            Hred = Hred[:,sampling_for_plot]
            P_full = plot(size=(400,400))
            for i in 1:length(to_plot)
                plot!(reverse(-xh), reverse(Hred[to_plot[i],:]'[:]), label="", size=(400,400),color=COL[i])
            end
            plot!(reverse(-xh),reverse(Hred[end,:]),label=L"Median~filter",color="black", linestyle=:dot, xlabel=L"Time~[ms]")
            plot!(xtickfontsize=12, ytickfontsize=12,legendfontsize=12,xlabelfontsize=16, ylabelfontsize=16)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(P_full, joinpath(folder_path, "Comp_Filter$(Name)_tanh.pdf"))
        end

        A = ifelse.(H[1:end-1,:] .> 0, 1, 0).*H[1:end-1,:]
        B = ifelse.(H[1:end-1,:] .< 0, 1, 0).*H[1:end-1,:]

        A1 = zeros(size(A,1),1)
        A2 = zeros(size(A,1),1)

        for i in 1:size(A,1)
            a = 0
            ind = 0
            for j in argmin(A[i,:]):size(A,2)
                if A[i,j]>0
                    a = A[i,j:end]
                    ind  = ind + j
                    break
                end 
            end
            if length(a)==1
                print(i)
            end
            for j in 1:length(a)
                if !(a[j]>0)
                    A1[i] = sum(a[1:j])
                    ind  = ind + j
                   break
                end 
            end
            # ok ici on a a1 
            for j in ind:size(A,2)
                if !(B[i,j]<0)
                    A2[i] = sum(B[i,ind:j])
                   break
                end 
            end
        end

        DC_spiking = DC[1:end-1]
        ISI_spiking = zeros(length(DC_spiking),1)
        ISI_s = zeros(length(DC_spiking),1)
        STD = zeros(length(ISI_spiking),1)
        ISI_Period = zeros(length(ISI_spiking),1)
        burstiness_bis = zeros(length(ISI_spiking),1)
        for i in 1:length(ISI_spiking)
            spike = read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/Spikesof$(Name)/spike_$(i).csv")[1:end-1]
            ISI = spike[2:end].-spike[1:end-1]
            if Name != NAME[1]
                ISI = sort(ISI)
                cutoff = (maximum(ISI)+minimum(ISI))/2
                for j in 1:length(ISI)
                    if ISI[j] > cutoff
                        ISI_s[i] = median(ISI[1:j-1])
                        ISI_e = ISI[j:end]
                        n_s_per_burst = (j-1)/length(ISI_e) 
                        burstiness_bis[i] = n_s_per_burst
                        ISI_Period[i] = median(ISI_e .+ (n_s_per_burst*ISI_s[i])).*dt
                        ISI_s[i] = ISI_s[i] .* dt
                        break
                    end
                end
            else
                ISI_spiking[i] = median(ISI).*dt
            end
            STD[i] = std(ISI)
        end
        # SCAT = scatter(ISI_spiking, DC_spiking, xlabel=L"Inter~Spike~Interval~[ms]", ylabel=L"Offset~component", color=COL, label="")
        # folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
        # savefig(SCAT, joinpath(folder_path, "ISI_vs_DC_$(Name).pdf"))

        path_folder = "SavingOfComputation/WithArthurFyonCodes_$(nt)/SaveInformationFitting/$(Name)"
        first_fb_pos_clean = zeros(N,1)
        first_fb_pos_DC = zeros(N,1)
        first_fb_pos_Formulae = zeros(N,1)

        second_fb_pos_clean = zeros(N,1)
        second_fb_pos_DC = zeros(N,1)

        for i in 1:N
            h = H[i,:]
            a = argmin(h)
            b = argmax(h)
            c = argmin(h[b:end])+b
            d = length(h)

            for j in a:b
                if h[j] > 0 
                    first_fb_pos_clean[i] = (j)*dt*sampling.step
                    break
                end
            end

            for j in a:b
                if h[j] + DC_spiking[i] > 0
                    first_fb_pos_DC[i] = (j)*dt*sampling.step
                    break
                end
            end

            for j in a:b
                if h[j] > (log(-log(1-(1-0.5^(1/100)))) - DC_spiking[i])
                    first_fb_pos_Formulae[i] = (j)*dt*sampling.step
                    break
                end
            end

            if Name!=NAME[1]
                for j in c:d
                    if h[j] > 0 
                        second_fb_pos_clean[i] = (j)*dt*sampling.step
                        break
                    end
                end

                for j in c:d
                    if h[j] + DC_spiking[i] > 0 
                        second_fb_pos_DC[i] = (j)*dt*sampling.step
                        break
                    end
                end
            end

        end
        pos = findall(x -> x == Name, NAME)[1]
        n = (pos-1)*500+1
        H_carac[n:n+N-1,1] = A1.*dt.*sampling.step
        H_carac[n:n+N-1,2] = A2.*dt.*sampling.step
        H_carac[n:n+N-1,3] = DC[1:end-1]
        if Name == NAME[1]
            H_carac[n:n+N-1,4] = ISI_spiking
        else
            H_carac[n:n+N-1,5] = ISI_Period
            H_carac[n:n+N-1,6] = ISI_s
        end
        H_carac[n:n+N-1,7] = STD
        if length(burstiness) != N
            burstiness = burstiness_bis
        end
        H_carac[n:n+N-1,8] = burstiness
        H_carac[n:n+N-1,9] = first_fb_pos_clean
        H_carac[n:n+N-1,10] = first_fb_pos_DC
        H_carac[n:n+N-1,11] = first_fb_pos_Formulae
        H_carac[n:n+N-1,12] = second_fb_pos_clean
        H_carac[n:n+N-1,13] = second_fb_pos_DC
        H_carac[n:n+N-1,14] = burstiness_bis
        H_carac[n:n+N-1,15] = minimum(H[1:end-1,:],dims=2)
        H_carac[n:n+N-1,16] = maximum(H[1:end-1,:],dims=2)
        GM = getindex.(argmax(H[1:end-1,:],dims=2),2)
        Gm = getindex.(argmin(H[1:end-1,:],dims=2),2)
        H_carac[n:n+N-1,17] = GM
        H_carac[n:n+N-1,18] = Gm
        H_carac[n:n+N-1,19] = minimum.(H[i, GM[i]:end] for i in 1:length(GM))
        H_carac[n:n+N-1,20] = zeros(N,1) .+ pos
        H_carac[n:n+N-1,21] = 1:1:500
    end

    for Name in NAME
        N = convert(Int,(read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/N_SIM/$(Name).csv")[1]) )

        n = (findall(x -> x == Name, NAME)[1]-1)*500+1

        ISI_spiking = H_carac[n:n+N-1,4]
        ISI_Period = H_carac[n:n+N-1,5]
        ISI_s = H_carac[n:n+N-1,6]
        STD = H_carac[n:n+N-1,7]
        burstiness = H_carac[n:n+N-1,8]
        first_fb_pos_clean = H_carac[n:n+N-1,9]
        first_fb_pos_DC = H_carac[n:n+N-1,10]
        first_fb_pos_Formulae = H_carac[n:n+N-1,11]
        second_fb_pos_clean = H_carac[n:n+N-1,12]
        second_fb_pos_DC = H_carac[n:n+N-1,13]

        if Name == NAME[1]
            col = []
            for i in 1:length(ISI_spiking)
                if STD[i] < 25
                    push!(col,"blue")
                else
                    push!(col,"red")
                end
            end
            mss = 5
            SCAT = scatter(ISI_spiking, first_fb_pos_clean, xlabel=L"Real~ISI~[ms]", ylabel=L"ISI~on~GLM~data~[ms]",color=col,label="",ms=mss,xlabelfontsize=16, ylabelfontsize=16,xtickfontsize=12, ytickfontsize=12)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(SCAT, joinpath(folder_path, "ISI_vs_ISIh_$(Name)_STD_clean.pdf"))

            SCAT = scatter(ISI_spiking, first_fb_pos_clean, xlabel=L"Real~ISI~[ms]", ylabel=L"ISI~on~GLM~data~[ms]",color="blue",label="",ms=mss,xlabelfontsize=16, ylabelfontsize=16,xtickfontsize=12, ytickfontsize=12)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(SCAT, joinpath(folder_path, "ISI_vs_ISIh_$(Name)_clean.pdf"))

            SCAT = scatter(ISI_spiking, first_fb_pos_DC, xlabel=L"Real~ISI~[ms]", ylabel=L"ISI~on~GLM~data~[ms]",color=col,label="",ms=mss,xlabelfontsize=16, ylabelfontsize=16,xtickfontsize=12, ytickfontsize=12)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(SCAT, joinpath(folder_path, "ISI_vs_ISIh_$(Name)_DC.pdf"))

            SCAT = scatter(ISI_spiking, first_fb_pos_Formulae, xlabel=L"Real~ISI~[ms]", ylabel=L"ISI~on~GLM~data~[ms]",color=col,label="",ms=mss,xlabelfontsize=16, ylabelfontsize=16,xtickfontsize=12, ytickfontsize=12)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(SCAT, joinpath(folder_path, "ISI_vs_ISIh_$(Name)_Formulae.pdf"))
        else
            mySalmon = RGB{Float64}(243. / 255., 124. / 255., 130. / 255.)
            myYellow  = RGB{Float64}(253. / 255., 211. / 255., 44. / 255.)
            mycmap = ColorScheme([mySalmon, myYellow]);
            my_col = mySalmon-myYellow
            diff_col = rescale_to_01(burstiness) .* my_col
            COL = diff_col .+ myYellow

            if nt == "STG"
                yl = (5,20)
                yll = (80,170)
            else
                yl = nothing
                yll = nothing
            end
            mss = 5
            SCAT = scatter(ISI_s, first_fb_pos_clean, xlabel=L"T_s~[ms]", ylabel=L"T_s~on~filter~h~[ms]",color=COL,label="",ylims=yl)
            plot!(ms=mss,xlabelfontsize=16, ylabelfontsize=16,xtickfontsize=12, ytickfontsize=12)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(SCAT, joinpath(folder_path, "Ts_$(Name)_clean.pdf"))

            SCAT = scatter(ISI_s, first_fb_pos_DC, xlabel=L"T_s~[ms]", ylabel=L"T_s~on~filter~h~[ms]",color=COL,label="",ylims=yl)
            plot!(ms=mss,xlabelfontsize=16, ylabelfontsize=16,xtickfontsize=12, ytickfontsize=12)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(SCAT, joinpath(folder_path, "Ts_$(Name)_DC.pdf"))

            SCAT = scatter(ISI_s, first_fb_pos_Formulae, xlabel=L"T_s~[ms]", ylabel=L"T_s~on~filter~h~[ms]",color=COL,label="",ylims=yl)
            plot!(ms=mss,xlabelfontsize=16, ylabelfontsize=16,xtickfontsize=12, ytickfontsize=12)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(SCAT, joinpath(folder_path, "Ts_$(Name)_Formulae.pdf"))

            SCAT = scatter(ISI_Period, second_fb_pos_clean, xlabel=L"T_b~[ms]", ylabel=L"T_b~on~filter~h~[ms]",color=COL,label="",ylims=yll)
            plot!(ms=mss,xlabelfontsize=16, ylabelfontsize=16,xtickfontsize=12, ytickfontsize=12)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(SCAT, joinpath(folder_path, "Tb_$(Name)_clean.pdf"))

            SCAT = scatter(ISI_Period, second_fb_pos_DC, xlabel=L"T_b~[ms]", ylabel=L"T_b~on~filter~h~[ms]",color=COL,label="")
            plot!(ms=mss,xlabelfontsize=16, ylabelfontsize=16,xtickfontsize=12, ytickfontsize=12)
            folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
            savefig(SCAT, joinpath(folder_path, "Tb_$(Name)_DC.pdf"))
        end
    end

    mss = 5
    N = convert(Int,(read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/N_SIM/$(NAME[1]).csv")[1]) )
    n = 501
    Ratio_A_1 =  H_carac[n:n+N-1,1] ./  H_carac[n:n+N-1,2]
    n = 1001
    Ratio_A_2 = H_carac[n:n+N-1,1] ./  H_carac[n:n+N-1,2]
    burst = H_carac[501:end,14]

    mySalmon = RGB{Float64}(243. / 255., 124. / 255., 130. / 255.)
    myYellow  = RGB{Float64}(253. / 255., 211. / 255., 44. / 255.)
    mycmap = ColorScheme([mySalmon, myYellow]);
    my_col = mySalmon-myYellow
    diff_col = rescale_to_01(log.(burst)) .* my_col
    COL = diff_col .+ myYellow;

    Ratio_A_1 = -Ratio_A_1
    Ratio_A_2 = -Ratio_A_2
    SCAT = scatter([0.6:((1.4-0.6)/(length(Ratio_A_1)-1)):1.4], Ratio_A_1,label="",color=COL[1:500])
    scatter!([1+0.6:((1.4-0.6)/(length(Ratio_A_2)-1)):1+1.4], Ratio_A_2,label="",color=COL[501:1000])
    annotate!(SCAT, 1, -0.05, text(L"Light~bursting", :center, 16)) 
    annotate!(SCAT, 2, -0.05, text(L"Strong~bursting", :center, 16))
    scatter!(xticks=[],ytickfontsize=12, ms=mss, ylabelfontsize=16, ylabel=L"β",bottom_margin=10Plots.mm)
    folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
    savefig(SCAT, joinpath(folder_path, "Ration_A1A2_allBURST.pdf"))

    NNN = []
    Correct_Name = ["Spiking", "Light bursting", "Strong bursting"]
    for i in 1:length(NAME)
        for j in 1:500
            push!(NNN,Correct_Name[i]) 
        end
    end
    mss = 5
    df = DataFrame(Offset=H_carac[:,3], Behaviour=NNN)
    using PlotlyJS
    Plot_DC = PlotlyJS.plot(df,x=:Behaviour,y=:Offset,kind="box")
    plot!(ms=mss,xlabelfontsize=16, ylabelfontsize=16,xtickfontsize=12, ytickfontsize=12)
    folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Figures"], "", "", 1)
    PlotlyJS.savefig(Plot_DC, joinpath(folder_path, "Comp_DC_value.pdf"))
    gr()

end


if classification_last
    # Distinction between spiking and bustring (10 and 1)
    P1 = H_carac[:,1]./ H_carac[:,16]
    P2 = H_carac[:,10]
    global SCAT = scatter()
    global SCATbis = scatter()
    SCATgeneral = scatter()
    COLOR = ["yellow","orange","red"]
    Correct_Name = [L"Spiking", L"Light~bursting", L"Strong~bursting"]
    for Name in NAME
        pos = findall(x -> x == Name, NAME)[1]
        n = (pos-1)*500+1
        ib = n:n+500-1
        global SCAT = scatter(SCAT,P2[ib],P1[ib],color=COLOR[pos],label=Correct_Name[pos])
        global SCATbis = scatter(SCATbis,P2[ib],P1[ib],color=COLOR[pos],label="")
    end
    mss = 5
    zone1 = [20,50]
    zone2 = [575,650]
    SCAT = scatter(SCAT, xlims=(zone1[1],zone1[2]),legend=:topleft,legendfontsize=12,ms=mss)
    ylabel!(latexstring("\$Ratio\$ \$of\$ \$the\$ \$integral\$ \$of\$ \$first\$ \n \$positive\$ \$feedback\$ \$and\$ \$its\$ \$top\$ \$value\$"), ylabelfontsize=16)
    SCATbis = scatter(SCATbis, xlims=(zone2[1],zone2[2]),yaxis=false,ms=mss)
    SCATgeneral = scatter(SCAT,SCATbis,layout = (1, 2),bottom_margin=10Plots.mm, right_margin=5Plots.mm, top_margin=10Plots.mm)
    annotate!(SCATgeneral, zone1[2], -100, text(L"~~~~~~~~~~~~Position~of~first~positive~feedback~[ms]", :center, 16))
    scatter!(ytickfontsize=12, xtickfontsize=12, ms=mss, ylabelfontsize=16,xlabelfontsize=16,legendfontsize=12)

    folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Classification_Fig"], "", "", 1)
    savefig(SCATgeneral, joinpath(folder_path, "Time_Int_posFB.pdf"))

    Reduction = zeros(1500,2)
    Reduction[:,1] = P1
    Reduction[:,2] = P2

    features = collect(Reduction')
    result = kmeans(features, 2)
    clusters = result.assignments;

    if ( sum(clusters[1:500] .== 1) > sum(clusters[1:500] .== 2) ) && ( sum(clusters[501:end] .== 1) > sum(clusters[501:end] .== 2) )
        println("Impossible to determine the familly")
        return 0
    end
    SCAT = scatter()
    COLOR = ["blue","green"]
    Correct_Name = [L"Spiking", L"Light~bursting", L"Strong~bursting"]
    scatter!(P2,P1,color=COLOR[clusters],label="")
    scatter!(xlabel=L"Position~of~first~positive~feedback~[ms]") #, ylabel=L"Integral~of~first~positive~feedback")
    ylabel!(latexstring("\$Ratio\$ \$of\$ \$the\$ \$integral\$ \$of\$ \$first\$ \n \$positive\$ \$feedback\$ \$and\$ \$its\$ \$top\$ \$value\$"), ylabelfontsize=16)
    scatter!(ytickfontsize=12, xtickfontsize=12, ms=mss, ylabelfontsize=16,xlabelfontsize=16, top_margin=10Plots.mm)
    folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Classification_Fig"], "", "", 1)
    savefig(SCAT, joinpath(folder_path, "Time_Int_posFB_CLUSTERS.pdf"))
    
    num_clus_spi = 1
    if ( sum(clusters[1:500] .== 1) < sum(clusters[1:500] .== 2) )
        num_clus_spi = num_clus_spi + 1
    end
    num_clus_burst = 3 - num_clus_spi

    Name_vect = []
    for i in 1:length(NAME)
        for j in 1:500
            push!(Name_vect, NAME[i]) 
        end
    end

    C1 = clusters[1:500]
    numC1 = num_clus_spi
    name_to_save = "Is_classified_as_SPIKING_but_is_BURSTING.pdf"
    Bad1 = built_bad_class(C1, numC1, name_to_save, Name_vect[1:500])
    if Bad1 != []
        save_data(["SavingOfComputation","WithArthurFyonCodes","WrongClassification","isBURSTING_accH"],"Class_error.csv",Bad1[:])
    end

    C1 = clusters[501:end]
    numC1 = num_clus_burst
    name_to_save = "Is_classified_as_BURSTING_but_is_SPIKING.pdf"
    Bad2 = built_bad_class(C1, numC1, name_to_save, Name_vect[501:end]) .+ 500
    if Bad2 != []
        save_data(["SavingOfComputation","WithArthurFyonCodes","WrongClassification","isSPIKING_accH"],"Class_error.csv",Bad2[:])
    end

    nb = sum(clusters .== num_clus_burst)
    ns = sum(clusters .== num_clus_spi)
    H_burst = zeros(nb, 21)
    H_sp = zeros(ns, 21)
    global j1 = 1
    global j2 = 1
    for i in 1:1500
        if clusters[i] == num_clus_burst
            H_burst[j1,:] = H_carac[i,:]
            global j1 = j1 + 1
        else 
            H_sp[j2,:] = H_carac[i,:]
            global j2 = j2 + 1
        end
    end

    to_plot_sp = randperm(ns)[1:10]
    global j = 1
    PLT = plot()
    for i in 1:length(to_plot_sp)
        ieme = convert(Int,to_plot_sp[i])
        name = NAME[ convert(Int, H_sp[ieme,20]) ]
        i_sp = convert(Int,H_sp[ieme,21])
        spike = read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/Spikesof$(name)/spike_$(i_sp).csv", 1)[1:end-1]
        for x in 1:convert(Int,round( length(spike)/2 ))
            if spike[x] == 1
                plot!([x,x].*dt, [j-0.35, j+0.35], label="", color="black")
            end
        end
        global j = j + 1
    end
    plot!(xlabel=L"Time~[ms]",ylabel=L"Sequence~number")
    plot!(xlabelfontsize=16,ytickfontsize=12,xtickfontsize=12,ylabelfontsize=16,yticks=1:length(to_plot_sp))
    folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Classification_Fig"], "", "", 1)
    savefig(PLT, joinpath(folder_path, "Spiking.pdf"))

    A2 = log.(10,-H_burst[:,2])

    features = collect(A2')
    result = kmeans(features, 2)
    clusters = result.assignments;

    SCAT = scatter()
    COLOR = ["blue","green"]
    Correct_Name = [L"Cluster~A", L"Cluster~B"]
    global x1 = 0.6:((1.4-0.6)/(sum(clusters.==1)-1)):1.4
    global x2 = (0.6:((1.4-0.6)/(sum(clusters.==2)-1)):1.4).+1
    Rg = randperm(length(A2))
    for i in Rg
        if clusters[i] == 1
            scatter!([x1[1]],[A2[i]],color=COLOR[clusters[i]],label="")
            if length(x1) != 1
                global x1 = x1[2:end]
            end
        end
        if clusters[i] == 2
            scatter!([x2[1]],[A2[i]],color=COLOR[clusters[i]],label="")
            if length(x2) != 1
                global x2 = x2[2:end]
            end
        end
    end 
    annotate!(SCAT, 1, 2.45, text(L"Cluster~A", :center, 16)) 
    annotate!(SCAT, 2, 2.45, text(L"Cluster~B", :center, 16)) 

    scatter!(ylabel=L"Integral~of~second~negative~feedback")
    scatter!(ytickfontsize=12, ms=mss, ylabelfontsize=16,legendfontsize=12,xticks=[],top_margin=10Plots.mm,bottom_margin=10Plots.mm)

    folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Classification_Fig"], "", "", 1)
    savefig(SCAT, joinpath(folder_path, "Int_negFB_CLUSTERS.pdf"))

    n1 = sum(clusters.==1)
    n2 = sum(clusters.==2)

    H_burst1 = zeros(n1,21)
    H_burst2 = zeros(n2,21)

    global j1 = 1
    global j2 = 1
    for i in 1:(n1+n2)
        if clusters[i] == 1
            H_burst1[j1,:] = H_burst[i,:]
            global j1 += 1
        else
            H_burst2[j2,:] = H_burst[i,:]
            global j2 += 1
        end
    end

    to_plot_1 = randperm(n1)[1:10]
    to_plot_2 = randperm(n2)[1:10]

    global j = 1
    PLT = plot()
    for i in 1:length(to_plot_1)
        ieme = convert(Int,to_plot_1[i])
        name = NAME[ convert(Int, H_burst1[ieme,20]) ]
        i_sp = convert(Int,H_burst1[ieme,21])
        spike = read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/Spikesof$(name)/spike_$(i_sp).csv", 1)[1:end-1]
        for x in 1:convert(Int,round( length(spike)/2 ))
            if spike[x] == 1
                plot!([x,x].*dt, [j-0.35, j+0.35], label="", color="black")
            end
        end
        global j = j + 1
    end
    plot!(xlabel=L"Time~[ms]",ylabel=L"Sequence~number")
    plot!(xlabelfontsize=16,ytickfontsize=12,xtickfontsize=12,ylabelfontsize=16,yticks=1:length(to_plot_sp))
    folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Classification_Fig"], "", "", 1)
    savefig(PLT, joinpath(folder_path, "Bursting_Case1.pdf"))

    global j = 1
    PLT = plot()
    for i in 1:length(to_plot_2)
        ieme = convert(Int,to_plot_2[i])
        name = NAME[ convert(Int, H_burst2[ieme,20]) ]
        i_sp = convert(Int,H_burst2[ieme,21])
        spike = read_data("SavingOfComputation/WithArthurFyonCodes_$(nt)/Spikesof$(name)/spike_$(i_sp).csv", 1)[1:end-1]
        for x in 1:convert(Int,round( length(spike)/2 ))
            if spike[x] == 1
                plot!([x,x].*dt, [j-0.35, j+0.35], label="", color="black")
            end
        end
        global j = j + 1
    end
    plot!(xlabel=L"Time~[ms]",ylabel=L"Sequence~number")
    plot!(xlabelfontsize=16,ytickfontsize=12,xtickfontsize=12,ylabelfontsize=16,yticks=1:length(to_plot_sp))
    folder_path = save_data(["SavingOfComputation","WithArthurFyonCodes_$(nt)","Classification_Fig"], "", "", 1)
    savefig(PLT, joinpath(folder_path, "Bursting_Case2.pdf"))

end


if same_b_diff_h
    x = [0, 0.1, 1, 1.1, 2, 2.1,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15].*100
    h1 =[0,  -1,-1, 0.8,   0.8, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0,   0 ]
    h2 =[0,  -1,-1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, -1, -1, -1, -1, -1, 0.4, 0.4, 0.4, 0,   0 ]
    PLT = plot()
    plot!(reverse(-[x[1],x[end]]), [0,0],label="", linestyle=:dot, color="black")
    plot!(-reverse(x),reverse(h1),color="blue", linewidth=3, label=L"Filter~shape~1")
    plot!(-reverse(x),reverse(h2),color="red", linewidth=2, label=L"Filter~shape~2")
    plot!(ylims=(-1.1,1.1), xlabel=L"Time~[ms]",xlabelfontsize=16,ytickfontsize=12,xtickfontsize=12,legendfontsize=12)
    folder_path = save_data(["SavingOfComputation","Fig"], "", "", 1)
    savefig(PLT, joinpath(folder_path, "Diff_h_same_b.pdf"))
end
