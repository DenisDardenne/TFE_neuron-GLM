







# Generate HM on different computer

using Plots
println("Start loading lib_fct.jl")
default(fmt = :png)    
include("lib_fct.jl");
println("End loading lib_fct.jl")

"""
    NOTE : 
        Some of the following 'if condition' are not used.
        But I decide to let it for your own use.
        As some 'if condition' are not used in the final version of the paper :
            "Modelling and classification of neuronal dynamics through Generalised Linear Models",
            I dont guaranty the good implementation. The checked 'if condition' are noted by a "G"
"""

# Generate HM on different computer
fct1 = false # G

# Investigate about the shape of the LL map, local min (due to numerical error)
fct2 = false # G

# Investigate about the shape of the LL map (n-dimension), local min (due to numerical error).
fct2bis = false


if fct1
    global Ts = [400 1100;
        400 1250;
        400 1100;
        400 1250;
        100 1250;
        350 1250;
        9250 15250;
        8250 14250;
        750 1050;
        200 1100; # 10
        1900 4000;
        5100 9100;
        600 1500;
        600 1500;
        1400 2200; # 15
        10 100;
        200 1100;
        300 2200;
        200 1100;
        200 1100; #20
        550 850;
        100 1000;
        100 10000];
    global Periods = [1000;
                    1000;
                    1000;
                    1000;
                    1000;
                    1000;
                    -1;
                    -1; # 8
                    -1;
                    -1;
                    -1;
                    -1;
                    -1;
                    -1;
                    -1;
                    -1;
                    -1;
                    -1;
                    -1;
                    -1; # 20
                    -1]

    cell = 4 # phasic bursting
    # cell = 1
    fct_fit = "LLfit"
    TB = ""
    n_tP = 1
    nbasK = 3
    nbasH = 12

    lk1 = ReentrantLock()

    I1, dt1, T1, title1, time1 = generate_stimulation(cell)
    Lm = convert(Int,(Ts[cell,2]+Periods[cell]*(n_tP-1))/dt1)
    if Lm <= length(I1)
        I1 = I1[1:Lm]
        time1 = time1[1:Lm]
        I_small = I1[1:convert(Int,(Ts[cell,2])/dt1)]
    else
        println("Error with the size of training period")
    end

    v1, u1, spikes1 = simulate_izhikevich(cell,I1,dt1);
    ~, ~, spikes1_small = simulate_izhikevich(cell,I_small,dt1);

    maxIter = 1000;  # max number of iterations for fitting, also used for maximum number of function evaluations(MaxFunEvals)
    tolFun = 1e-12;  # function tolerance for fitting
    solver = [NewtonTrustRegion()]

    nb = 3:14

    global V_LL = zeros(length(nb),length(nb))
    for nbasK in nb
        nkt = 100; # number of ms in stim filter
        kbasprs = Dict(
            :neye => 0,  # number of "identity" basis vectors near time of spike
            :ncos => nbasK,  # number of raised-cosine vectors to use
            :kpeaks => [0.1, round(nkt / 1.2)],  # position of first and last bump relative to identity bumps
            :b => 10  # how nonlinear to make spacings (larger values make it more linear)
        )
        Threads.@threads for nbasH in nb
            # basis functions for post-spike kernel
            ihbasprs = Dict(
                :ncols => nbasH,  # number of basis vectors for post-spike kernel
                :hpeaks => [0.1, 100],  # peak location for the first and last vectors, in ms
                :b => 10,  # how nonlinear to make spacings (larger values make it more linear)
                :absref => 1  # absolute refractory period, in ms
            )
            ~, ~, ~, ~, ~, ~, trace = fit_glm(I1,spikes1,dt1,nkt,kbasprs,ihbasprs,[],maxIter,tolFun, solver[1], TB, fct_fit);
            lock(lk1) do
                global V_LL[nbasK-2,nbasH-2] = trace[end].value
            end
        end
    end

    V_LL_prime_normed = (V_LL./V_LL[1,1])'
    HM = heatmap(nb,nb,V_LL_prime_normed,xlabel=L"n_k", ylabel=L"n_h",xticks=nb,yticks=nb,color=:buda)
    plot!(xlabelfontsize=16,ytickfontsize=12,xtickfontsize=12,ylabelfontsize=16,right_margin=10Plots.mm)

    savefig(HM,"ScoreLLfit_accordingNumOfBasis_Wind.pdf")

end

if fct2
    for cell in [4]
        global Ts = [400 1100;
            400 1250;
            400 1100;
            400 1250;
            100 1250;
            350 1250;
            9250 15250;
            8250 14250;
            750 1050;
            200 1100; # 10
            1900 4000;
            5100 9100;
            600 1500;
            600 1500;
            1400 2200; # 15
            10 100;
            200 1100;
            300 2200;
            200 1100;
            200 1100; #20
            550 850;
            100 1000;
            100 10000];
        global Periods = [1000;
            1000;
            1000;
            1000;
            1000;
            1000;
            -1;
            -1; # 8
            -1;
            -1;
            -1;
            -1;
            -1;
            -1;
            -1;
            -1;
            -1;
            -1;
            -1;
            -1; # 20
            -1]

        nbasK = 6
        nbasH = 7
        fct_fit = "LLfit"
        TB = ""
        n_tP = 1
        nkt = 100; # number of ms in stim filter
        kbasprs = Dict(
            :neye => 0,  # number of "identity" basis vectors near time of spike
            :ncos => nbasK,  # number of raised-cosine vectors to use
            :kpeaks => [0.1, round(nkt / 1.2)],  # position of first and last bump relative to identity bumps
            :b => 10  # how nonlinear to make spacings (larger values make it more linear)
        )
        # basis functions for post-spike kernel
        ihbasprs = Dict(
            :ncols => nbasH,  # number of basis vectors for post-spike kernel
            :hpeaks => [0.1, 100],  # peak location for the first and last vectors, in ms
            :b => 10,  # how nonlinear to make spacings (larger values make it more linear)
            :absref => 1  # absolute refractory period, in ms
        )

        I1, dt1, T1, title1, time1 = generate_stimulation(cell)
        Lm = convert(Int,(Ts[cell,2]+Periods[cell]*(n_tP-1))/dt1)
        if Lm <= length(I1)
            I1 = I1[1:Lm]
            time1 = time1[1:Lm]
            I_small = I1[1:convert(Int,(Ts[cell,2])/dt1)]
        else
            println("Error with the size of training period")
        end

        v1, u1, spikes1 = simulate_izhikevich(cell,I1,dt1);
        ~, ~, spikes1_small = simulate_izhikevich(cell,I_small,dt1);
        maxIter = 1000;  # max number of iterations for fitting, also used for maximum number of function evaluations(MaxFunEvals)
        tolFun = 1e-12;  # function tolerance for fitting
        solver = [NewtonTrustRegion()]
        k, h, dc, prs, kbasis, hbasis, trace = fit_glm(I1,spikes1,dt1,nkt,kbasprs,ihbasprs,[],maxIter,tolFun, solver[1], TB, fct_fit);

        y = spikes1
        x = I1
        xconvki = zeros(size(y, 1), nbasK)
        yconvhi = zeros(size(y, 1), nbasH)

        for knum in 1:nbasK
            xconvki[:, knum] = sameconv(x, kbasis[:, knum])
        end

        for hnum in 1:nbasH
            yconvhi[:, hnum] = sameconv(y, reverse(hbasis[:, hnum]))
        end

        kprs = prs[1:nbasK];
        dt = 0.1
        refreshRate = 1000 / dt
        dc_prs = prs[end]
        h3_prs = prs[nbasK+2]

        step = 0.01
        r = 0.9:step:1.1
        range_ll_dc = r
        range_ll_h = r

        LL = zeros(length(range_ll_h),length(range_ll_dc))

        hprs_ref = prs[nbasK+1:end-1];
        xconvk = xconvki*kprs

        for ih in range_ll_h
            hprs = hprs_ref.*ih
            yconvh = yconvhi*hprs
            dx = findfirst(x -> x == ih, range_ll_h)
            for idc in range_ll_dc
                xconvk_dc = xconvk .+ (idc*prs[end]);
                g = (xconvk_dc + yconvh);
                lambda = exp.(g)
                dy = findfirst(x -> x == idc, range_ll_dc)
                A = (-y'*g)[1]
                B = dt*sum(lambda)/refreshRate
                LL[dx,dy] = A + B;  # negative log likelihood
                if LL[dx,dy] > 1
                    LL[dx,dy] = log(10,LL[dx,dy]) + 1
                end
            end
        end
        P1 = heatmap(range_ll_h,range_ll_dc,LL,color=:buda,xlabel=L"C~/~C^*", ylabel=L"h~/~h^*",xtickfontsize=12, ytickfontsize=12, xlabelfontsize=16, ylabelfontsize=16, right_margin=10Plots.mm)
        x_offset = (range_ll_h[end]-range_ll_h[1])/10
        side = -1
        Loc_mini = retrun_list_of_min(LL)
        default_colors = palette(:prism, length(Loc_mini))
        i = 0
        for i in 1:length(Loc_mini)
            x = Loc_mini[i][1]
            dx = range_ll_h[x]
            y = Loc_mini[i][2]
            dy = range_ll_dc[y]
            value = LL[x, y]
            scatter!([dy], [dx], m=:o, ms=3, mc=default_colors[i], msw=0, label=false, alpha=0.8) #on P1
            annotate!(dy, dx+(side*x_offset), text(string(round(value, digits=2)), color=default_colors[i], :bold, 10))
            side *= -1
        end
        title!("")

        savefig(P1,"Local_solutions_$(cell)_K$(nbasK)_H$(nbasH)_with_r_$(r[1])_$(step)_$(r[end]).pdf")
    end

end

if fct2bis
    global Ts = [400 1100;
    400 1250;
    400 1100;
    400 1250;
    100 1250;
    350 1250;
    9250 15250;
    8250 14250;
    750 1050;
    200 1100; # 10
    1900 4000;
    5100 9100;
    600 1500;
    600 1500;
    1400 2200; # 15
    10 100;
    200 1100;
    300 2200;
    200 1100;
    200 1100; #20
    550 850;
    100 1000;
    100 10000];
    global Periods = [1000;
        1000;
        1000;
        1000;
        1000;
        1000;
        -1;
        -1; # 8
        -1;
        -1;
        -1;
        -1;
        -1;
        -1;
        -1;
        -1;
        -1;
        -1;
        -1;
        -1; # 20
        -1]

    cell = 4
    nbasK = 5
    nbasH = 13
    fct_fit = "LLfit"
    TB = ""
    n_tP = 1
    nkt = 100; # number of ms in stim filter
    kbasprs = Dict(
        :neye => 0,  # number of "identity" basis vectors near time of spike
        :ncos => nbasK,  # number of raised-cosine vectors to use
        :kpeaks => [0.1, round(nkt / 1.2)],  # position of first and last bump relative to identity bumps
        :b => 10  # how nonlinear to make spacings (larger values make it more linear)
    )
    # basis functions for post-spike kernel
    ihbasprs = Dict(
    :ncols => nbasH,  # number of basis vectors for post-spike kernel
    :hpeaks => [0.1, 100],  # peak location for the first and last vectors, in ms
    :b => 10,  # how nonlinear to make spacings (larger values make it more linear)
    :absref => 1  # absolute refractory period, in ms
    )

    I1, dt1, T1, title1, time1 = generate_stimulation(cell)
    Lm = convert(Int,(Ts[cell,2]+Periods[cell]*(n_tP-1))/dt1)
    if Lm <= length(I1)
        I1 = I1[1:Lm]
        time1 = time1[1:Lm]
        I_small = I1[1:convert(Int,(Ts[cell,2])/dt1)]
    else
        println("Error with the size of training period")   
    end

    v1, u1, spikes1 = simulate_izhikevich(cell,I1,dt1);
    ~, ~, spikes1_small = simulate_izhikevich(cell,I_small,dt1);
    maxIter = 10000;  # max number of iterations for fitting, also used for maximum number of function evaluations(MaxFunEvals)
    tolFun = 1e-12;  # function tolerance for fitting
    solver = [NewtonTrustRegion()]
    k, h, dc, prs, kbasis, hbasis, trace = fit_glm(I1,spikes1,dt1,nkt,kbasprs,ihbasprs,prs_ms,maxIter,tolFun, solver[1], TB, fct_fit);

    y = spikes1
    x = I1
    xconvki = zeros(size(y, 1), nbasK)
    yconvhi = zeros(size(y, 1), nbasH)

    for knum in 1:nbasK
    xconvki[:, knum] = sameconv(x, kbasis[:, knum])
    end

    for hnum in 1:nbasH
    yconvhi[:, hnum] = sameconv(y, reverse(hbasis[:, hnum]))
    end

    kprs = prs[1:nbasK];
    dt = 0.1
    refreshRate = 1000 / dt
    dc_prs = prs[end]
    # h3_prs = prs[nbasK+2]
    h_prs = prs[nbasK+1:end-1]
    hprs_ref = prs[nbasK+1:end-1];

    xconvk = xconvki*kprs

    # M =  10*maximum(abs.(vcat(h_prs,dc_prs)))
    M =  1.5*(maximum(abs.(vcat(h_prs,dc_prs)))+5)
    #m = 2*(minimum(abs.(vcat(h_prs,dc_prs)))+10)
    stepp = M/3
    #r = -(M+stepp):stepp:(m+stepp)
    r = -M:stepp:M

    global range_ll_k1 = prs[1]-2*(abs(prs[1]/2)):abs(prs[1]/2):prs[1]+2*(abs(prs[1]/2)) # r # -130:1:-120
    global range_ll_k2 = prs[2]-2*(abs(prs[2]/2)):abs(prs[2]/2):prs[2]+2*(abs(prs[2]/2)) # r # -7:0.5:-2
    global range_ll_k3 = prs[3]-2*(abs(prs[3]/2)):abs(prs[3]/2):prs[3]+2*(abs(prs[3]/2)) # r #  0:0.1:1.5
    global range_ll_k4 = prs[4]-2*(abs(prs[4]/2)):abs(prs[4]/2):prs[4]+2*(abs(prs[4]/2)) # r #  0:0.1:1.5
    global range_ll_k5 = prs[5]-2*(abs(prs[5]/2)):abs(prs[5]/2):prs[5]+2*(abs(prs[5]/2)) # r #  0:0.1:1.5
    global range_ll_h1 = prs[nbasK+1]-2*(abs(prs[nbasK+1]/2)):abs(prs[nbasK+1]/2):prs[nbasK+1]+2*(abs(prs[nbasK+1]/2)) # r #  0:0.1:1.5
    global range_ll_h2 = prs[nbasK+2]-2*(abs(prs[nbasK+2]/2)):abs(prs[nbasK+2]/2):prs[nbasK+2]+2*(abs(prs[nbasK+2]/2)) # r #  0:0.1:1.5
    global range_ll_h3 = prs[nbasK+3]-2*(abs(prs[nbasK+3]/2)):abs(prs[nbasK+3]/2):prs[nbasK+3]+2*(abs(prs[nbasK+3]/2)) # r #  0:0.1:1.5
    global range_ll_dc = prs[end]-2*(abs(prs[end]/2)):abs(prs[end]/2):prs[end]+2*(abs(prs[end]/2)) # r # -5:0.5:0

    azerty = -1:1:1
    global range_ll_k1 = prs[1]-1:1:prs[1]+1
    global range_ll_k2 = prs[2]-1:1:prs[2]+1
    global range_ll_k3 = prs[3]-1:1:prs[3]+1
    global range_ll_k4 = prs[4]-1:1:prs[4]+1
    global range_ll_k5 = prs[5]-1:1:prs[5]+1
    global range_ll_h1 = prs[nbasK+1]-1:1:prs[nbasK+1]+1
    global range_ll_h2 = prs[nbasK+2]-1:1:prs[nbasK+2]+1
    global range_ll_h3 = prs[nbasK+3]-1:1:prs[nbasK+3]+1
    global range_ll_dc = prs[end]-1:1:prs[end]+1

    prs_ms = [-2.5419128904391255,
    1.7503194046602797,
    2.3915127633178463,
    -1.9435013744720089,
    -1.05207652241387,
    -6.796750953543903,
    -5.088315157668355,
    -7.691684184817795,
    -5.495044475240522,
    -5.559767264885814,
    -4.01352910151917,
    -2.27752921673703,
    -2.3097985730551907,
    -0.5090832482039199,
    1.439606888286826,
    -1.2607856578693497,
    0.011081635309056613,
    0.00945382951868787,
    -2.460487737988234,]

    prec = 1
    azerty = -2*prec:prec:2*prec
    global range_ll_k1 = azerty # .+ prs[1]
    global range_ll_k2 = azerty # .+ prs[2]
    global range_ll_k3 = azerty # .+ prs[3]
    global range_ll_k4 = azerty # .+ prs[4]
    global range_ll_k5 = azerty # .+ prs[5]
    global range_ll_h1 = 1:1:1 # azerty # .+ prs[6]
    global range_ll_h2 = azerty # .+ prs[7]
    global range_ll_h3 = azerty # .+ prs[8]
    global range_ll_dc = azerty # .+ prs[9]

    prec = 1
    azerty = -prec:prec:prec
    global range_ll_k1 = azerty .+ prs_ms[1]
    global range_ll_k2 = azerty .+ prs_ms[2]
    global range_ll_k3 = azerty .+ prs_ms[3]
    global range_ll_k4 = azerty .+ prs_ms[4]
    global range_ll_k5 = azerty .+ prs_ms[5]
    global range_ll_h1 = azerty .+ prs_ms[6]
    global range_ll_h2 = azerty .+ prs_ms[7]
    global range_ll_h3 = azerty .+ prs_ms[8]
    global range_ll_dc = azerty .+ prs_ms[9]

    # range_ll_dc = -5:1:5
    # range_ll_h1 = -50:5:-10
    # range_ll_h2 = 0:1:15
    # range_ll_h3 = -5:1:5
    # range_ll_h4 = r
    # range_ll_h5 = r
    # range_ll_h6 = r
    # range_ll_h7 = r

    lk1 = ReentrantLock()
    lk2 = ReentrantLock()

    global gain = 0

    for _ in 1:1
        # LLfull = zeros(length(range_ll_h1),length(range_ll_h2),length(range_ll_h3),length(range_ll_h4),length(range_ll_h5),length(range_ll_h6),length(range_ll_dc));
        # LLfull = zeros(length(range_ll_h1),length(range_ll_h2),length(range_ll_h3),length(range_ll_dc));
        # LLfull = zeros(length(range_ll_k1),length(range_ll_k2),length(range_ll_k3),length(range_ll_h1),length(range_ll_h2),length(range_ll_h3),length(range_ll_dc));
        LLfull = zeros(length(range_ll_k1),length(range_ll_k2),length(range_ll_k3),length(range_ll_k4),length(range_ll_k5),length(range_ll_h1),length(range_ll_h2),length(range_ll_h3),length(range_ll_dc));
        #LL = zeros(length(range_ll),1)

        # Param = range_ll_k1,range_ll_k2,range_ll_k3,range_ll_h1,range_ll_h2,range_ll_h3,range_ll_dc
        Param = range_ll_k1,range_ll_k2,range_ll_k3,range_ll_k4,range_ll_k5,range_ll_h1,range_ll_h2,range_ll_h3,range_ll_dc
        # Ind_Param = 1:length(Param[1]),1:length(Param[2]),1:length(Param[3]),1:length(Param[4]),1:length(Param[5]),1:length(Param[6]),1:length(Param[7]) # ,1:length(Param[8])
        Ind_Param = 1:length(Param[1]),1:length(Param[2]),1:length(Param[3]),1:length(Param[4]),1:length(Param[5]),1:length(Param[6]),1:length(Param[7]),1:length(Param[8]),1:length(Param[9])

        First = minimum(CartesianIndices(Ind_Param))
        L = length(First)

        @time begin
            center = CartesianIndices(Ind_Param)[1]*2
            @showprogress Threads.@threads for Cind in CartesianIndices(Ind_Param)
                kprs = zeros(nbasK)
                hprs = zeros(nbasH)
                dcvalue = 0
                lock(lk1) do
                    for i in 1:nbasK
                        kprs[i] = Param[i][Cind[i]]
                    end
                    for i in nbasK+1:L-1
                        hprs[i-nbasK] = Param[i][Cind[i]]
                    end
                    dcvalue = Param[end][Cind[L]]
                end
                g = (((xconvki*kprs) .+ dcvalue) + (yconvhi*hprs));
                AB = (-y'*g)[1] + dt*sum(exp.(g))/refreshRate
                lock(lk2) do
                    if AB > 1
                        LLfull[Cind] = log(10,AB) + 1
                    else
                        LLfull[Cind] = AB
                    end
                end
            end
        end

        Loc_mini = findall(==(minimum(LLfull)), LLfull)[1]

        println(Loc_mini)
        println(LLfull[Loc_mini])

            Loc_mini
            global P = zeros(L)
            for i in 1:L
                global P[i] = Param[i][Loc_mini[i]]
            end
            global vll = LLfull[Loc_mini]

            global range_ll_k1 = upd_range(P[1], range_ll_k1)
            global range_ll_k2 = upd_range(P[2], range_ll_k2)
            global range_ll_k3 = upd_range(P[3], range_ll_k3)
            global range_ll_k4 = upd_range(P[4], range_ll_k4)
            global range_ll_k5 = upd_range(P[5], range_ll_k5)
            # global range_ll_h1 = upd_range(P[nbasK+1], range_ll_h1)
            global range_ll_h2 = upd_range(P[nbasK+2], range_ll_h2)
            global range_ll_h3 = upd_range(P[nbasK+3], range_ll_h3)
        #   global range_ll_h7 = upd_range(P[7], range_ll_h7)
            global range_ll_dc = upd_range(P[end], range_ll_dc)

            println(vll)
            println(P)
            println(vll-gain)
            global gain = vll
    end

    function upd_range(p, range_i)
        # if p <= range_i[1] || p >= range_i[end]
        #     pas = (range_i[2] - range_i[1])*1 #  = abs(p)/2
        #     pas = abs(p)/2
        #     range_i = p-2*pas:pas:p+2*pas
        # else
        #     pas = range_i[2] - range_i[1]  
        #     range_i = p-1*pas:pas/2:p+1*pas
        #     pas = range_i[2] - range_i[1]  
        #     if pas <= 1
        #         range_i = p-1:1:p+1
        #     end
        # end
        # return range_i
        gap = 1
        return p-gap:gap:p+gap
    end

    function retrun_list_of_min(LLfull) # not the plateau

        list_conv = []
        list_conv_LL = []
        LLfull_M1 = zeros(size(LLfull)) .+ Inf;
        LLfull_M2 = zeros(size(LLfull)) ;
        Un = CartesianIndices(LLfull)[1]
        Last = CartesianIndices(LLfull)[end]
        ndim = length(Un)
        for Cind in CartesianIndices(LLfull)
            global Condi = true
            for i in 1:ndim
                if Cind[i] == Un[i] || Cind[i] == Last[i]
                    global Condi = false
                end
            end
            if Condi
                Vois = Cind-Un:Cind+Un
                if LLfull[Cind] != Inf && sum(LLfull[Cind].<=LLfull[Vois]) == 3^ndim
                    LLfull_M1[Cind] = 0
                    M2 = LLfull_M2[Vois]
                    V = LLfull[Cind].== LLfull[Vois]
                    if sum(V) >= 2
                        if maximum(M2) == 0
                            LLfull_M2[Cind] = 2
                            push!(list_conv, Cind)
                            push!(list_conv_LL, LLfull[Cind])
                        else
                            LLfull_M2[Cind] = 1
                            LLfull_M1[Cind] = Inf
                        end
                    else
                        LLfull_M2[Cind] = 2
                        push!(list_conv, Cind)
                        push!(list_conv_LL,LLfull[Cind])
                    end
                end
            end
        end
        println(length(list_conv))
        LLfull_M1_prec = -1
        while LLfull_M1_prec != LLfull_M1
            println("Round")
            LLfull_M1_prec = copy(LLfull_M1)
            for Cind in CartesianIndices(LLfull)
                Condi = true
                for i in 1:ndim
                    if Cind[i] == Un[i] || Cind[i] == Last[i]
                        Condi = false
                    end
                end
                if Condi
                    if LLfull_M2[Cind] == 1 || LLfull_M2[Cind] == 2 
                        Vois = Cind-Un:Cind+Un
                        A = LLfull_M1[Vois]
                        B = LLfull[Vois]
                        for a in 1:length(A)
                            if ( A[a] == Inf && B[a] == LLfull[Cind] )
                                LLfull_M1[Cind] = Inf 
                            end
                        end
                    end
                end
            end
        end
        return findall(==(0), LLfull_M1)
    end

    Loc_mini = retrun_list_of_min(LLfull)
    if Loc_mini == []
        Loc_mini = findall(==(minimum(LLfull)), LLfull)
        if length(Loc_mini) != 1
            Loc_mini = Loc_mini[1]
        end
    else
        Loc_mini = Loc_mini[1]
    end

    Loc_mini
    P = zeros(L)
    for i in 1:L
        P[i] = Param[i][Loc_mini[i]]
    end
    LLfull[Loc_mini]
    P;
end