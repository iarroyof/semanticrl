using PyCall
using ScikitLearn
using CSV
using DataFrames
using IterTools
using HDF5
using JLD

@sk_import feature_extraction.text: TfidfVectorizer


#=
TODO: 1. To add density function (Gaussian and Exponential)
TODO: 2. PMF probabilities are the expected value over the sample space
TODO: 3. Verify wheter this programs is faster than python version (or how to 
         optimize, cartesian products are really needed?): 778.174950 seconds 
         (24.45 G allocations: 733.674 GiB, 66.92% gc time) (13 min, 92 steps,
         320 samples)
TODO: 4. To add Entropy, Conditional Entropy, Mutual Information and Divergence
TODO: 5. Probability surfaces inducing subordered set-valued outcomes
=#

function build_tokenizer(ngramr::Tuple)
    text_preprocessor = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngramr)
    tokenizer = text_preprocessor.build_analyzer()
    return tokenizer
end


function build_set_RVs(df::DataFrame, ngramer::Tuple)

    tokenizer = build_tokenizer(ngramer)

    X = Set.(tokenizer.(df.Column2))
    Y = Set.(tokenizer.(df.Column3))
    Z = Set.(tokenizer.(df.Column4))

    return (X, Y, Z)
end


function build_set_vocabs(x_set::Array{Set{String}, 1},
        y_set::Array{Set{String}, 1}, z_set::Array{Set{String}, 1})
    omega_x = []
    omega_y = []
    omega_z = []

    for x in x_set
        if x in omega_x
            continue
        else
            push!(omega_x, x)
        end
    end

    for y in y_set
        if y in omega_y
            continue
        else
            push!(omega_y, y)
        end
    end

    for z in z_set
        if z in omega_z
            continue
        else
            push!(omega_z, z)
        end
    end

    return (omega_x, omega_y, omega_z)
end


function intersect_lens(Oa, B)
    # Generate cartesian product (Oa x B) for all \omega in Oa and b in B
    cart_prod = product(Oa, B)
    lengths = length.(broadcast(a -> intersect(a[1], a[2])[1],
                                eachrow(collect(cart_prod))))
    return zip(collect(cart_prod), lengths)
end


function compute_intersects(omega_x, omega_y, omega_z, X, Y, Z)

    ints_x = intersect_lens(omega_x, X)
    ints_y = intersect_lens(omega_y, Y)
    ints_z = intersect_lens(omega_z, Z)

    ints_xy = intersect_lens(omega_x, Y)
    ints_yz = intersect_lens(omega_y, Z)
    ints_zx = intersect_lens(omega_z, X)

    return (ints_x, ints_y, ints_z, ints_xy, ints_yz, ints_zx)
end


function compute_proba(omega_a, ints_a, omega_b, ints_b)
    set_hash = []
    rv_mem = []
    
    for b in omega_b
        partition = 0
        bint_lens = [c for ((a_, b_), c) in ints_b if b == a_]

        for a in omega_a
            aint_lens = [c for ((a_, b_), c) in ints_a if a == a_]
            a_dot_b = aint_lens'bint_lens
            push!(set_hash, ((a, b), a_dot_b))
            partition += a_dot_b
        end
        push!(rv_mem, (b, partition))
    end
    set_hash = Dict(set_hash)
    rv_mem = Dict(rv_mem)
    P_AB = Dict([((a, b), set_hash[(a, b)]/rv_mem[b])
                    for (a, b) in product(omega_a, omega_b)])

    return P_AB
end


function compute_probas(ixx, iyy, izz, ixy, iyz, izx, ohm_x, ohm_y, ohm_z)
    P_xx = compute_proba(ohm_x, ixx, ohm_x, ixx)
    P_yy = compute_proba(ohm_y, iyy, ohm_y, iyy)
    P_zz = compute_proba(ohm_z, izz, ohm_z, izz)

    P_ygx = compute_proba(ohm_y, iyy, ohm_x, ixx)
    P_zgy = compute_proba(ohm_z, izz, ohm_y, iyy)
    P_zgx = compute_proba(ohm_z, izz, ohm_x, ixx)

    return Dict("P_X" => P_xx, "P_Y" => P_yy, "P_Z" => P_zz,
                "P_Y|X" => P_ygx, "P_Z|Y" => P_zgy, "P_Z|X" => P_zgx)
end


function process_step(X_set, Y_set, Z_set)
    omega_x, omega_y, omega_z = build_set_vocabs(X_set, Y_set, Z_set)
    xx, yy, zz, xy, yz, zx = compute_intersects(
                            omega_x, omega_y, omega_z, X_set, Y_set, Z_set)
    dists = compute_probas(xx, yy, zz, xy, yz, zx, omega_x, omega_y, omega_z)

    return dists
end


function main()
    input_tsv_triplets = "/almac/ignacio/semanticrl/data/dis_train.txt.oie"
    step_size = 320
    inputs = []
    @time begin
    for rows in Iterators.partition(CSV.Rows(input_tsv_triplets, delim='\t', header=false), step_size)
        df = DataFrame(rows)
        push!(inputs, build_set_RVs(df, (1, 4)))
    end
    println("Chunks created... ")
    end
    n_steps = length(inputs)
    results = [Dict() for _ in 1:n_steps]
    s = @sprintf "Ready to process a total of %5.1f steps of %5.1f samples each..." n_steps step_size;
    println(s)

    @time begin
    Threads.@threads for (i, (X_set, Y_set, Z_set)) in collect(enumerate(inputs))
        @time begin
        dists = process_step(X_set, Y_set, Z_set, i)
        results[i] = dists
        s = @sprintf "Step %5.1f" i;
        println(s)
        end
    end
    println("All finished..")
    end
    
    save("results.jld", "results", results)

end

main()
#results = process_step()
#N = 10
#results = [Dict() for _ in 1:N]
#@distributed for i in 1:N
#    dist = process_step()
#    results[i] = dist
#end
#print(results[1])

