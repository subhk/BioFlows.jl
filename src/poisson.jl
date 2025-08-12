using LinearAlgebra

"""
Solve Laplacian(p) = rhs on a uniform grid with homogeneous Neumann BCs
using a simple conjugate gradient on the 5/7-point stencil.

2D: p size (nx,ny), dx,dy given
3D: p size (nx,ny,nz), dx,dy,dz given
"""
function solve_poisson!(p::AbstractArray{T,2}, rhs::AbstractArray{T,2}, dx, dy; maxiter=2000, tol=1e-6) where {T}
    nx, ny = size(p)
    A_mul!(y, x) = begin
        @inbounds for j in 1:ny
            for i in 1:nx
                c = -2/(dx^2) - 2/(dy^2)
                v = c * x[i,j]
                v += (i>1 ? x[i-1,j]/(dx^2) : x[i,j]/(dx^2))
                v += (i<nx ? x[i+1,j]/(dx^2) : x[i,j]/(dx^2))
                v += (j>1 ? x[i,j-1]/(dy^2) : x[i,j]/(dy^2))
                v += (j<ny ? x[i,j+1]/(dy^2) : x[i,j]/(dy^2))
                y[i,j] = v
            end
        end
        y
    end
    cg!(p, rhs, A_mul!; maxiter, tol)
end

function solve_poisson!(p::AbstractArray{T,3}, rhs::AbstractArray{T,3}, dx, dy, dz; maxiter=4000, tol=1e-6) where {T}
    nx, ny, nz = size(p)
    A_mul!(y, x) = begin
        @inbounds for k in 1:nz
            for j in 1:ny
                for i in 1:nx
                    c = -2/(dx^2) - 2/(dy^2) - 2/(dz^2)
                    v = c * x[i,j,k]
                    v += (i>1 ? x[i-1,j,k]/(dx^2) : x[i,j,k]/(dx^2))
                    v += (i<nx ? x[i+1,j,k]/(dx^2) : x[i,j,k]/(dx^2))
                    v += (j>1 ? x[i,j-1,k]/(dy^2) : x[i,j,k]/(dy^2))
                    v += (j<ny ? x[i,j+1,k]/(dy^2) : x[i,j,k]/(dy^2))
                    v += (k>1 ? x[i,j,k-1]/(dz^2) : x[i,j,k]/(dz^2))
                    v += (k<nz ? x[i,j,k+1]/(dz^2) : x[i,j,k]/(dz^2))
                    y[i,j,k] = v
                end
            end
        end
        y
    end
    cg!(p, rhs, A_mul!; maxiter, tol)
end

# Minimal CG on arrays without forming A.
function cg!(x, b, A_mul!; maxiter=1000, tol=1e-6)
    r = similar(x); Ap = similar(x); p = similar(x)
    fill!(x, 0)
    copyto!(r, b)
    copyto!(p, r)
    rsold = dot(r, r)
    for it in 1:maxiter
        A_mul!(Ap, p)
        α = rsold / dot(p, Ap)
        @. x += α * p
        @. r -= α * Ap
        rsnew = dot(r, r)
        if sqrt(rsnew) < tol
            break
        end
        β = rsnew / rsold
        @. p = r + β * p
        rsold = rsnew
    end
    return x
end

