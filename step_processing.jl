using PyCall
using ScikitLearn
using CSV
using DataFrames
using IterTools

@sk_import feature_extraction.text: TfidfVectorizer


function build_tokenizer(ngramr::Tuple)
    text_preprocessor = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngramr)
    tokenizer = text_preprocessor.build_analyzer()
    return tokenizer
end


function build_set_RVs(input_tsv::String, ngramer::Tuple)
    df = DataFrame(CSV.File(input_tsv, delim='\t', header=false))

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


function main()
    input_tsv_triplets = "/almac/ignacio/semanticrl/data/dis_train_320.txt.oie"
    X_set, Y_set, Z_set = build_set_RVs(input_tsv_triplets, (1, 4))
    @time begin
    omega_x, omega_y, omega_z = build_set_vocabs(X_set, Y_set, Z_set)
    print("Vocabs: ")
    end #   0.039684 seconds (105.95 k allocations: 2.175 MiB)
    @time begin
    xx, yy, zz, xy, yz, zx = compute_intersects(
                            omega_x, omega_y, omega_x, X_set, Y_set, Z_set)
    print("Ints: ")
    end #  1.370609 seconds (4.18 M allocations: 178.062 MiB, 4.65% gc time)

    @time begin
    dists = compute_probas(xx, yy, zz, xy, yz, zx, omega_x, omega_y, omega_z)
    print("Probas: ")
    end 

    # Show example
    y = omega_y[1]
    P_Yx = [dists["P_Y|X"][(y, x)] for x in omega_y]
    println(P_Xy)
    println(sum(P_Xy))
end    


main()
