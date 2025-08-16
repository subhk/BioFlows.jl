"""
Post-Processing and Analysis for 2D Cylinder Flow Results

This script provides tools to analyze the output from the cylinder flow simulation,
including force coefficients, vortex shedding frequency, and flow visualization.
"""

using NCDatasets
using Statistics
using Printf
using FFTW

"""
Load force coefficient data from NetCDF file
"""
function load_force_coefficients(filename::String)
    """
    Load drag and lift coefficients from simulation output.
    
    Returns:
    - t: time array
    - Cd: drag coefficient array  
    - Cl: lift coefficient array
    """
    
    if !isfile(filename)
        error("File $filename not found. Make sure simulation completed successfully.")
    end
    
    NCDataset(filename, "r") do ds
        t = ds["time"][:]
        Cd = ds["drag_coefficient"][:]
        Cl = ds["lift_coefficient"][:]
        
        @printf "Loaded force data: %d time points\n" length(t)
        @printf "Time range: %.2f to %.2f seconds\n" minimum(t) maximum(t)
        
        return t, Cd, Cl
    end
end

"""
Analyze vortex shedding frequency using FFT
"""
function analyze_shedding_frequency(t, Cl; Re=100.0)
    """
    Compute vortex shedding frequency and Strouhal number.
    """
    
    # Remove initial transient (first 25% of data)
    n_transient = Int(floor(length(t) * 0.25))
    t_steady = t[n_transient:end]
    Cl_steady = Cl[n_transient:end]
    
    @printf "Analyzing steady portion: %.2f to %.2f seconds\n" t_steady[1] t_steady[end]
    
    # Compute power spectral density
    dt = t_steady[2] - t_steady[1]
    fs = 1.0 / dt  # Sampling frequency
    
    # Apply window to reduce spectral leakage
    window = hanning(length(Cl_steady))
    Cl_windowed = Cl_steady .* window
    
    # FFT
    Cl_fft = fft(Cl_windowed)
    freqs = fftfreq(length(Cl_steady), fs)
    
    # Power spectral density (one-sided)
    n_half = div(length(freqs), 2)
    freqs_pos = freqs[1:n_half]
    psd = abs.(Cl_fft[1:n_half]).^2
    
    # Find dominant frequency (excluding DC component)
    idx_max = argmax(psd[2:end]) + 1
    f_shedding = freqs_pos[idx_max]
    
    # Compute Strouhal number: St = f * D / U
    D = 0.4  # Cylinder diameter
    U = 1.0  # Inlet velocity
    St = f_shedding * D / U
    
    @printf "Vortex shedding analysis:\n"
    @printf "  Shedding frequency: %.4f Hz\n" f_shedding
    @printf "  Strouhal number: %.4f\n" St
    
    # Compare with empirical correlation for circular cylinder
    # St ≈ 0.198 * (1 - 19.7/Re) for 40 < Re < 200
    if 40 < Re < 200
        St_empirical = 0.198 * (1 - 19.7/Re)
        error_percent = abs(St - St_empirical) / St_empirical * 100
        @printf "  Empirical St: %.4f (error: %.1f%%)\n" St_empirical error_percent
    end
    
    return f_shedding, St, freqs_pos, psd
end

"""
Compute time-averaged drag coefficient
"""
function analyze_drag_coefficient(t, Cd)
    """
    Compute mean drag coefficient and statistics.
    """
    
    # Remove initial transient
    n_transient = Int(floor(length(t) * 0.25))
    Cd_steady = Cd[n_transient:end]
    
    Cd_mean = mean(Cd_steady)
    Cd_std = std(Cd_steady)
    Cd_min = minimum(Cd_steady)
    Cd_max = maximum(Cd_steady)
    
    @printf "Drag coefficient analysis:\n"
    @printf "  Mean Cd: %.4f ± %.4f\n" Cd_mean Cd_std
    @printf "  Range: %.4f to %.4f\n" Cd_min Cd_max
    
    # Compare with literature values for Re=100
    Cd_literature = 1.33  # Approximate value for Re=100
    error_percent = abs(Cd_mean - Cd_literature) / Cd_literature * 100
    @printf "  Literature Cd: %.2f (error: %.1f%%)\n" Cd_literature error_percent
    
    return Cd_mean, Cd_std
end

"""
Load and analyze flow field data
"""
function analyze_flow_field(filename::String, time_index::Int=-1)
    """
    Load and analyze flow field at specified time.
    time_index = -1 loads the last time step.
    """
    
    NCDataset(filename, "r") do ds
        # Get dimensions
        x = ds["x"][:]
        z = ds["z"][:]  # Vertical coordinate in XZ plane
        times = ds["time"][:]
        
        if time_index == -1
            time_index = length(times)
        end
        
        t = times[time_index]
        @printf "Analyzing flow field at t = %.2f s\n" t
        
        # Load velocity components
        u = ds["u"][:, :, time_index]  # x-velocity
        w = ds["w"][:, :, time_index]  # z-velocity (vertical)
        p = ds["p"][:, :, time_index]  # pressure
        
        # Compute derived quantities
        velocity_magnitude = sqrt.(u.^2 + w.^2)
        
        # Compute vorticity (∂w/∂x - ∂u/∂z)
        dx = x[2] - x[1]
        dz = z[2] - z[1]
        
        # Simple finite difference for vorticity
        dwdx = zeros(size(w))
        dudz = zeros(size(u))
        
        # Interior points
        for i in 2:size(w,1)-1
            dwdx[i, :] = (w[i+1, :] - w[i-1, :]) / (2*dx)
        end
        
        for j in 2:size(u,2)-1
            dudz[:, j] = (u[:, j+1] - u[:, j-1]) / (2*dz)
        end
        
        vorticity = dwdx - dudz
        
        # Flow statistics
        u_max = maximum(velocity_magnitude)
        p_min = minimum(p)
        p_max = maximum(p)
        vort_max = maximum(abs.(vorticity))
        
        @printf "Flow field statistics:\n"
        @printf "  Max velocity: %.3f m/s\n" u_max
        @printf "  Pressure range: %.3f to %.3f Pa\n" p_min p_max
        @printf "  Max vorticity: %.3f s⁻¹\n" vort_max
        
        return x, z, u, w, p, velocity_magnitude, vorticity
    end
end

"""
Check AMR refinement effectiveness
"""
function analyze_amr_refinement(filename::String)
    """
    Analyze adaptive mesh refinement data.
    """
    
    NCDataset(filename, "r") do ds
        if "refinement_level" in keys(ds)
            refinement_levels = ds["refinement_level"][:, :, end]  # Last time step
            
            # Count cells at each refinement level
            level_counts = Dict()
            for level in unique(refinement_levels)
                level_counts[level] = count(==(level), refinement_levels)
            end
            
            total_cells = length(refinement_levels)
            
            @printf "AMR refinement analysis:\n"
            @printf "  Total cells: %d\n" total_cells
            for level in sort(collect(keys(level_counts)))
                count = level_counts[level]
                percentage = count / total_cells * 100
                @printf "  Level %d: %d cells (%.1f%%)\n" level count percentage
            end
            
            # Compute refinement efficiency
            base_cells = level_counts[0]
            refined_cells = total_cells - base_cells
            refinement_ratio = refined_cells / total_cells * 100
            
            @printf "  Refinement ratio: %.1f%%\n" refinement_ratio
            
            return refinement_levels, level_counts
        else
            println("No AMR data found in file")
            return nothing, nothing
        end
    end
end

"""
Comprehensive analysis of cylinder flow results
"""
function comprehensive_analysis(base_filename::String; Re::Float64=100.0)
    """
    Perform complete analysis of cylinder flow simulation results.
    """
    
    println("="^60)
    println("Comprehensive Analysis of Cylinder Flow Results")
    println("="^60)
    
    # File names
    forces_file = base_filename * "_forces.nc"
    flow_file = base_filename * "_0001.nc"  # First flow field file
    
    try
        # 1. Force coefficient analysis
        println("\n1. Loading and analyzing force coefficients...")
        t, Cd, Cl = load_force_coefficients(forces_file)
        
        # Drag analysis
        println("\n2. Drag coefficient analysis...")
        Cd_mean, Cd_std = analyze_drag_coefficient(t, Cd)
        
        # Vortex shedding analysis
        println("\n3. Vortex shedding frequency analysis...")
        f_shed, St, freqs, psd = analyze_shedding_frequency(t, Cl; Re=Re)
        
        # 4. Flow field analysis
        if isfile(flow_file)
            println("\n4. Flow field analysis...")
            x, z, u, w, p, vel_mag, vorticity = analyze_flow_field(flow_file)
        else
            println("\n4. Flow field file not found: $flow_file")
        end
        
        # 5. AMR analysis
        if isfile(flow_file)
            println("\n5. AMR refinement analysis...")
            ref_levels, level_counts = analyze_amr_refinement(flow_file)
        end
        
        # Summary
        println("\n" * "="^60)
        println("ANALYSIS SUMMARY")
        println("="^60)
        @printf "Reynolds number: %.1f\n" Re
        @printf "Mean drag coefficient: %.4f ± %.4f\n" Cd_mean Cd_std
        @printf "Strouhal number: %.4f\n" St
        @printf "Shedding frequency: %.4f Hz\n" f_shed
        
        println("\nAnalysis completed successfully!")
        
        # Recommendations
        println("\nRecommendations for further analysis:")
        println("- Plot Cd and Cl vs time to visualize periodic behavior")
        println("- Create contour plots of vorticity to see vortex street")
        println("- Animate velocity magnitude to show vortex shedding")
        println("- Compare results with experimental/literature data")
        
        return true
        
    catch e
        println("Analysis failed: $e")
        println("Make sure simulation completed and output files exist")
        return false
    end
end

"""
Create simple visualization commands (pseudo-code for plotting)
"""
function visualization_examples()
    """
    Provide examples of how to visualize the results.
    """
    
    println("\nVisualization Examples (requires plotting package):")
    println("="^50)
    
    println("\n# Load and plot force coefficients")
    println("using Plots")
    println("t, Cd, Cl = load_force_coefficients(\"cylinder_flow_2d_forces.nc\")")
    println("plot(t, [Cd Cl], label=[\"Cd\" \"Cl\"], xlabel=\"Time (s)\", ylabel=\"Force Coefficient\")")
    
    println("\n# Create vorticity contour plot")
    println("x, z, u, w, p, vel_mag, vorticity = analyze_flow_field(\"cylinder_flow_2d_0001.nc\")")
    println("contour(x, z, vorticity', levels=20, xlabel=\"x (m)\", ylabel=\"z (m)\", title=\"Vorticity\")")
    
    println("\n# Plot velocity magnitude with streamlines")
    println("contourf(x, z, vel_mag', xlabel=\"x (m)\", ylabel=\"z (m)\", title=\"Velocity Magnitude\")")
    println("# Add cylinder outline")
    println("θ = 0:0.1:2π")
    println("x_cyl = 2.0 .+ 0.2.*cos.(θ)")
    println("z_cyl = 2.0 .+ 0.2.*sin.(θ)")
    println("plot!(x_cyl, z_cyl, color=:black, linewidth=2)")
    
    println("\n# FFT analysis of lift coefficient")
    println("f_shed, St, freqs, psd = analyze_shedding_frequency(t, Cl)")
    println("plot(freqs[2:end], psd[2:end], xlabel=\"Frequency (Hz)\", ylabel=\"PSD\", yscale=:log10)")
end

# Run comprehensive analysis if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("BioFlow.jl: Cylinder Flow Analysis Tool")
    
    if length(ARGS) >= 1
        base_filename = ARGS[1]
        Re = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 100.0
        comprehensive_analysis(base_filename; Re=Re)
    else
        println("Usage: julia analyze_cylinder_results.jl <base_filename> [Reynolds_number]")
        println("Example: julia analyze_cylinder_results.jl cylinder_flow_2d 100")
        
        # Show visualization examples
        visualization_examples()
    end
end

export load_force_coefficients, analyze_shedding_frequency, analyze_drag_coefficient,
       analyze_flow_field, comprehensive_analysis