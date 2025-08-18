#!/usr/bin/env julia

# Simple test to check what's happening with RigidBodyCollection

println("Testing BioFlow.jl imports...")

try
    # Try to use BioFlows without precompilation
    @eval begin
        # Set up the path
        push!(LOAD_PATH, ".")
        
        # Include the module directly without precompilation
        include("src/BioFlows.jl")
        
        # Import the module
        using .BioFlows
        
        println("✓ BioFlows module loaded successfully!")
        
        # Test specific exports
        println("Testing exports...")
        
        # Test RigidBodyCollection
        try
            collection = RigidBodyCollection()
            println("✓ RigidBodyCollection() works!")
        catch e
            println("✗ RigidBodyCollection() failed: $e")
        end
        
        # Test Circle
        try
            circle = Circle(0.2, [2.0, 2.0])
            println("✓ Circle() works!")
        catch e
            println("✗ Circle() failed: $e")
        end
        
        # Test add_body!
        try
            collection = RigidBodyCollection()
            circle = Circle(0.2, [2.0, 2.0])
            add_body!(collection, circle)
            println("✓ add_body!() works!")
            println("  Bodies in collection: $(collection.n_bodies)")
        catch e
            println("✗ add_body!() failed: $e")
        end
        
    end
    
catch e
    println("✗ Failed to load BioFlows: $e")
    
    # Try to diagnose the issue
    println("\nDiagnosing the issue...")
    
    # Check if files exist
    if isfile("src/BioFlows.jl")
        println("✓ src/BioFlows.jl exists")
    else
        println("✗ src/BioFlows.jl missing")
    end
    
    if isfile("src/bodies/rigid_bodies.jl")
        println("✓ src/bodies/rigid_bodies.jl exists")
    else
        println("✗ src/bodies/rigid_bodies.jl missing")
    end
    
    # Check if we can at least include the rigid bodies file
    try
        include("src/core/types.jl")
        include("src/bodies/rigid_bodies.jl")
        println("✓ Can include rigid_bodies.jl directly")
        
        # Test RigidBodyCollection directly
        collection = RigidBodyCollection()
        println("✓ RigidBodyCollection works when included directly")
    catch e2
        println("✗ Cannot include rigid_bodies.jl: $e2")
    end
end