using Plots
default(fmt = :png)    
include("lib_fct.jl");
using LaTeXStrings



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


cell_num = [1,2,3,4]
folder_path = "SavingOfComputation/Figure"

for cell in cell_num
    I, dt, ~, title1, time = generate_stimulation(cell)
    a = convert(Int, Ts[cell,1]/dt )
    b = convert(Int, Ts[cell,2]/dt )
    I_small = I[a:b]
    time_small = time[a:b]
    v, ~, ~ = simulate_izhikevich(cell,I_small,dt);
    P = plot(time_small, v, label="", xlabel=L"Time~[ms]", ylabel=L"V_m~[mV]",xlabelfontsize=16, ylabelfontsize=16, xtickfontsize=12, ytickfontsize=12)
    adr = joinpath(folder_path, "Main_pattern_$(cell).pdf")
    println(adr)
    savefig(P, adr)
end
















# Generation of 4-7-10 basis for h-filter
dt=0.1
for nbasH in [4,7,10]
    ihbasprs = Dict(
        :ncols => nbasH,  # number of basis vectors for post-spike kernel
        :hpeaks => [0.1, 100],  # peak location for the first and last vectors, in ms
        :b => 10,  # how nonlinear to make spacings (larger values make it more linear)
        :absref => 1  # absolute refractory period, in ms
    )
    ht, hbas, hbasis = makeBasis_PostSpike(ihbasprs, dt)
    hbasis = vcat(zeros(1, ihbasprs[:ncols]), hbasis)  # enforce causality
    P = plot()
    l1 = size(hbasis,1)
    for i in 1:size(hbasis,2)
        plot!(-dt*l1:dt:-dt, reverse(hbasis[:,i]),linewidth=3, label="")
    end
    plot!(xlabel=L"Time~[ms]", ylabel=L"Intensity",xtickfontsize=12, ytickfontsize=12,xlabelfontsize=16, ylabelfontsize=16)
    savefig(P, joinpath("SavingOfComputation","Figure", "Basis_$(nbasH).pdf"))
end

