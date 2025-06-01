using Test
using SparseBase
using LinearAlgebra

@testset "SparseBase.jl Tests" begin
    
    @testset "Basic Functionality" begin
        @test SparseBase.novalue isa SparseBase.NoValue
        @test SparseBase.ColMajor() isa SparseBase.StorageOrder
        @test SparseBase.RowMajor() isa SparseBase.StorageOrder
        @test SparseBase.RuntimeOrder() isa SparseBase.StorageOrder
