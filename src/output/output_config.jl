"""
    NetCDFConfig (generic output configuration)

Generic output configuration reused by JLD2 writer. Kept the name for minimal
changes across the codebase.
"""
struct NetCDFConfig
    filename::String
    max_snapshots_per_file::Int
    save_mode::Symbol
    time_interval::Float64
    iteration_interval::Int
    save_flow_field::Bool
    save_body_positions::Bool
    save_force_coefficients::Bool
    reference_velocity::Float64
    flow_direction::Vector{Float64}
    enable_deflate::Bool
    deflate_level::Int
    shuffle_filter::Bool
    function NetCDFConfig(filename::String;
                         max_snapshots_per_file::Int = 100,
                         save_mode::Symbol = :both,
                         time_interval::Float64 = 0.1,
                         iteration_interval::Int = 10,
                         save_flow_field::Bool = true,
                         save_body_positions::Bool = true,
                         save_force_coefficients::Bool = true,
                         reference_velocity::Float64 = 1.0,
                         flow_direction::Vector{Float64} = [1.0, 0.0],
                         enable_deflate::Bool = true,
                         deflate_level::Int = 4,
                         shuffle_filter::Bool = true)
        if !(save_mode in [:time_interval, :iteration_interval, :both])
            error("save_mode must be :time_interval, :iteration_interval, or :both")
        end
        new(filename, max_snapshots_per_file, save_mode, time_interval, iteration_interval,
            save_flow_field, save_body_positions, save_force_coefficients,
            reference_velocity, flow_direction, enable_deflate, deflate_level, shuffle_filter)
    end
end

