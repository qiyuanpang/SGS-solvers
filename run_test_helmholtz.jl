using MPI

NPROCS = [parse(Int64, ARGS[1])]
Ns = [parse(Int64, ARGS[2])]
f = "test_helmholtz.jl"
nfail = 0
blas_nths = 1
# @info "Running SpMM tests using MPI on 2D processor grids"
for N in Ns
    for nprocs in NPROCS
        try
            # run(`julia $(joinpath(@__DIR__, f)) $nprocs $N`)
            # run(`$(Base.julia_cmd()) $(joinpath(@__DIR__, f1)) $N $sqnprocs $what`)
            mpiexec() do cmd
                # run(`$cmd --mca opal_warn_on_missing_libcuda 0 -n $nprocs --map-by NUMA:PE=$blas_nths $(Base.julia_cmd()) $(joinpath(@__DIR__, f)) $N`)
                run(`$cmd -n $nprocs --map-by NUMA:PE=$blas_nths $(Base.julia_cmd()) $(joinpath(@__DIR__, f)) $N`)
            end
            Base.with_output_color(:green,stdout) do io
                println(io,"\tSUCCESS: $f")
            end
        catch ex
            Base.with_output_color(:red,stderr) do io
                println(io,"\tError: $f")
                showerror(io,ex,backtrace())
            end
            # nfail += 1
        end

    end
end

