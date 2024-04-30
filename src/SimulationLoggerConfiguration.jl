module SimulationLoggerConfiguration
    using Format
    using TimerOutputs
    using Logging, LoggingExtras
    using Printf
    using Dates
    using InteractiveUtils

    export SimulationLogger, generate_format_string, InitializeLogger, LogStep, LogFinal

    """
        function generate_format_string(values)

    Function to dynamically generate a format string based on values

    # Parameters
    - `values`: values to be formatted

    # Return
    - `format_str`: formatted string

    # Example
    ```julia
    format_string = generate_format_string(values)
    ```
    """
    
    function generate_format_string(values)
        # Calculate the display length for each value
        lengths = [length(string(value)) for value in values]
        
        # Optionally, add extra padding
        padding = 10 #maximum(lengths)  # Adjust padding as needed
        lengths = [len + padding for len in lengths]
        
        # Build format specifiers for each length
        format_specifiers = ["%-$(len)s" for len in lengths]
        
        # Combine into a single format string
        format_str = join(format_specifiers, " ")
        
        return format_str
    end

    """
        struct SimulationLogger

    SimulationLogger is a struct containing the parameters to output a log during the simulation.

    # Fields
    - `LoggerIo::IOStream`: Logger stream where log messages are written to.
    - `Logger::FormatLogger`: Logger that contains all logging information
    - `ValuesToPrint::String`: String that will be printed.
    - `ValuesToPrintC::String`: C-String that will be printed.
    - `CurrentDate::DateTime`: CurrentDate.
    - `CurrentDataStr::String`: LoggerMetaData string.

    # Example
    ```julia
    SimLogger = SimulationLogger(SimMetaDataDamBreak.SaveLocation)
    ```
    """

    struct SimulationLogger
        LoggerIo::IOStream
        Logger::FormatLogger
        FormatStr::String
        ValuesToPrint::String
        ValuesToPrintC::String
        CurrentDate::DateTime
        CurrentDataStr::String

        """
            function SimulationLogger(SaveLocation::String)

        SimulationLogger constructor

        # Parameters
        - `SaveLocation::String`: String where logger should write log files to.

        # Example
        ```julia
        SimLogger = SimulationLogger(SimMetaDataDamBreak.SaveLocation)
        ```
        """

        function SimulationLogger(SaveLocation::String)
            io_logger = open(SaveLocation * "/" * "SimulationOutput.log", "w")
            logger    = FormatLogger(io_logger::IOStream) do io, args
                # Write the module, level and message only
                # println(io, args._module, " | ", "[", args.level, "] ", args.message)
                println(io, args.message)
            end

            values        = ("PART [-]", "PartTime [s]", "TotalSteps [-] ", "Steps  [-] ", "Run Time [s]", "Time/Sec [-]", "Remaining Time [Date]")
            values_eq     = map(x -> repeat("=", length(x)), values)
            format_string = generate_format_string(values)

            ValuesToPrint  = @. $join(cfmt(format_string, values))
            ValuesToPrintC = @. $join(cfmt(format_string, values_eq))

            # This should not be hardcoded here.
            CurrentDate    = now()
            CurrentDataStr = Dates.format(CurrentDate, "dd-mm-yyyy HH:MM:SS")

            new(io_logger, logger, format_string, ValuesToPrint, ValuesToPrintC, CurrentDate, CurrentDataStr)
        end
    end

    """
        function InitializeLogger(SimLogger,SimConstants,SimMetaData)

    SimulationLogger constructor

    # Parameters
    - `SimLogger`: Logger that handles the output of the logger file.
    - `SimConstants`: Simulation SimConstants
    - `SimMetaData`: Simulation Meta data

    # Example
    ```julia
    InitializeLogger(SimLogger,SimConstants,SimMetaData)
    ```
    """

    function InitializeLogger(SimLogger,SimConstants,SimMetaData)
        with_logger(SimLogger.Logger) do
            @info sprint(InteractiveUtils.versioninfo)
            @info SimConstants
            @info SimMetaData
            
            # Print the formatted date and time
            @info "Logger Start Time: " * SimLogger.CurrentDataStr

            @info @. SimLogger.ValuesToPrint
            @info @. SimLogger.ValuesToPrintC
        end
    end

    """
        function LogStep(SimLogger,SimMetaData, HourGlass)

    logger output at each simulation step

    # Parameters
    - `SimLogger`: Logger that handles the output of the logger file.
    - `SimMetaData`: Simulation Meta data
    - `HourGlass`: Keeps track of execution time.

    # Example
    ```julia
    LogStep(SimLogger, SimMetaData, HourGlass)
    ```
    """

    function LogStep(SimLogger, SimMetaData, HourGlass)
        with_logger(SimLogger.Logger) do
            PartNumber               = "Part_" * lpad(SimMetaData.OutputIterationCounter,4,"0")
            PartTime                 = string(@sprintf("%-.6f", SimMetaData.TotalTime))
            PartTotalSteps           = string(SimMetaData.Iteration)
            CurrentSteps             = string(SimMetaData.Iteration - SimMetaData.StepsTakenForLastOutput)
            TimeUptillNow            = string(@sprintf("%-.3f",TimerOutputs.tottime(HourGlass)/1e9))
            TimePerPhysicalSecond    = string(@sprintf("%-.2f", TimerOutputs.tottime(HourGlass)/1e9 / SimMetaData.TotalTime))

            SecondsToFinish          = (SimMetaData.SimulationTime - SimMetaData.TotalTime) * (TimerOutputs.tottime(HourGlass)/1e9 / SimMetaData.TotalTime)
            ExpectedFinishTime       = now() + Second(ceil(Int,SecondsToFinish))
            ExpectedFinishTimeString = Dates.format(ExpectedFinishTime, "dd-mm-yyyy HH:MM:SS")

            @info @. $join(cfmt(SimLogger.FormatStr, (PartNumber, PartTime, PartTotalSteps,  CurrentSteps, TimeUptillNow, TimePerPhysicalSecond, ExpectedFinishTimeString)))
        end
    end

    """
        function LogFinal(SimLogger,HourGlass)

    logger output at the end of the simulation run

    # Parameters
    - `SimLogger`: Logger that handles the output of the logger file.
    - `HourGlass`: Keeps track of execution time.

    # Example
    ```julia
    LogFinal(SimLogger, HourGlass)
    ```
    """
    
    function LogFinal(SimLogger, HourGlass)
        with_logger(SimLogger.Logger) do
            # Get the current date and time
            current_time = now()
            # Format the current date and time
            formatted_time = "Simulation finished at: " * Dates.format(current_time, "dd-mm-yyyy HH:MM:SS")

            @info formatted_time
            @info "Simulation took " * @sprintf("%-.2f", TimerOutputs.tottime(HourGlass)/1e9) * "[s]"
            show(SimLogger.LoggerIo, HourGlass,sortby=:name)
            @info "\n Sorted by time \n"
            show(SimLogger.LoggerIo, HourGlass)
        end
    end

end