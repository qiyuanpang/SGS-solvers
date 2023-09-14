using LinearAlgebra
using MPI

function jacobi_mpi(Q, x0, y1, b, itermax, rank, dvd, comm)
    cputime = Dict("J"=>0.0, "invJ"=>0.0, "mainloop"=>0.0, "Allgatherv!"=>0.0)
    n = dvd[rank+1]
    a0, a1, a2 = 0, 1, 0
    s = 0

    cputime["J"] = @elapsed begin
    transition = 0
    for i in 1:length(dvd)
        if i != rank+1
            transition += norm(Q[:,s+1:s+dvd[i]])
        end
        s += dvd[i]
    end
    J = Q[:, sum(dvd[1:rank])+1:sum(dvd[1:rank])+n] + transition * Matrix{Float64}(I, n, n)
    end
    
    cputime["invJ"] = @elapsed begin
    invJ = inv(J)
    end

    iter = 1
    x1 = zeros(size(x0))
    y1_gather = Array{Float64}(undef, (sum(dvd), 1))
    _counts = vcat([1 for i in 1:length(dvd[:])]', dvd')
    y1_gather_vbuf = VBuffer(y1_gather, vec(prod(_counts, dims=1)))
    cputime["mainloop"] = @elapsed begin
    while iter <= itermax
        cputime["Allgatherv!"] += @elapsed begin
        MPI.Allgatherv!(y1, y1_gather_vbuf, comm)
        end
        x1 = y1 + invJ * (b - Q * y1_gather)
        a2 = (1.0+sqrt(1.0+4.0*a1*a1)) / 2.0
        y2 = (1+a0/a2) * x1 - a0/a2 * x0

        y1 .= y2
        a1, a0 = a2, a1
        x0 .= x1

        iter += 1
    end
    end

    cputime["totally"] = cputime["J"] + cputime["invJ"] + cputime["mainloop"]

    return x1, cputime
end