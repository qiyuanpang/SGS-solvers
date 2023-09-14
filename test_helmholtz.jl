using LinearAlgebra
using MPI
using Printf

include("./jacobi.jl")

function split_count(N::Integer, n::Integer)
    q,r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
comm_size = MPI.Comm_size(comm)

if true
    N = parse(Int64, ARGS[1])
    itermax = 10
    repeats = 10

    dvd = split_count(N, comm_size)
    dx = 1.0 / (N - 1)
    Q = zeros(dvd[rank+1], N)
    for gl_i in sum(dvd[1:rank])+1:sum(dvd[1:rank])+dvd[rank+1]
        lc_i = gl_i - sum(dvd[1:rank])
        if gl_i == 1
            Q[lc_i, 1], Q[lc_i, 2] = 2.0/dx/dx, -1.0/dx/dx
        elseif gl_i == N
            Q[lc_i, N-1], Q[lc_i, N] = -1.0/dx/dx, 2.0/dx/dx
        else
            Q[lc_i, gl_i-1], Q[lc_i, gl_i], Q[lc_i, gl_i+1] = -1.0/dx/dx, 2.0/dx/dx, -1.0/dx/dx
        end 
    end
    # F = eigen(Q)
    # @printf("min eig: %.2e max eig: %.2e \n", minimum(F.values), maximum(F.values))
    
    if rank == 0
        @printf("------------- Test helmholtz ------------ \n")
    end
    cputime = Dict()
    for jj in 1:repeats
        x0 = randn(dvd[rank+1], 1)
        y1 = zeros(size(x0))
        y1 .= x0
        b = randn(dvd[rank+1], 1)
        sol, cputime_jj = jacobi_mpi(Q, x0, y1, b, itermax, rank, dvd, comm)

        if jj == 1
            for (key, val) in cputime_jj
                cputime[key] = val
            end
        else
            if cputime_jj["totally"] < cputime["totally"]
                for (key, val) in cputime_jj
                    cputime[key] = val
                end
            end
        end
    end
    
    if rank == 0
        @printf("Node: %i \n", comm_size)
        @printf("N: %i itermax: %i \n", N, itermax)
        @printf("time totally: %.2e \n", cputime["totally"])
        @printf("    time J: %.2e \n", cputime["J"])
        @printf("    time invJ: %.2e \n", cputime["invJ"])
        @printf("    time mainloop: %.2e \n", cputime["mainloop"])
        @printf("        time Allgatherv!: %.2e \n", cputime["Allgatherv!"])
        @printf("\n\n")
    end

end


GC.gc()
MPI.Finalize()