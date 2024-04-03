module ProduceVTP
    export ExportVTP, ConvertHDFtoVTP, SaveHDF5!, HDFtoVTP, OpenForWriteH5

    #https://www.analytech-solutions.com/analytech-solutions/blog/binary-io.html
    using XML
    using XML: Document, Declaration, Element, Text
    using StaticArrays
    using HDF5

    ### Functions=================================================
    # Function to create a DataArray element for VTK files
    function create_data_array_element(name::String, data::AbstractVector{T}, offset::Int) where T
        # Create the DataArray elements
        dataarray = Element("DataArray")
        
        # Set attributes based on the input vector's type
        dataarray.attributes["type"]               = string(eltype(first(data)))
        dataarray.attributes["Name"]               = name  
        dataarray.attributes["NumberOfComponents"] = string(Int(sizeof(first(data))/sizeof(eltype(first(data)))))
        dataarray.attributes["format"]             = "appended"
        dataarray.attributes["offset"]             = string(offset)
        return dataarray
    end

    ###===========================================================
    # Points         = [SVector{3,Float64}(1,2,3), SVector{3,Float64}(4,5,6)]
    # Kernel         = Float64.([100, 200]) 
    # KernelGradient = [SVector{3,Float64}(-1,1,0), SVector{3,Float64}(1,-1,0)]
    # N              = 6195
    # Points         = rand(SVector{3,Float64},N) * 10
    # Kernel         = rand(Float64,N) * 1000
    # KernelGradient = rand(SVector{3,Float64},N) * 100

    ExportVTP(filename,points) = ExportVTP(filename,points,nothing)
    function ExportVTP(filename::String, points, variable_names, args...)
            # Generate the XML document and then put in some fixed values
            xml_doc = Document(Declaration(version=1.0,encoding="utf-8"))
            vtk_file = Element("VTKFile")
            vtk_file.attributes["type"]        = "PolyData"
            vtk_file.attributes["version"]     = "1.0"
            vtk_file.attributes["byte_order"]  = "LittleEndian"
            vtk_file.attributes["header_type"] = "UInt64"

            # PolyData is the main section, filling it out
            polydata  = Element("PolyData")
            piece     = Element("Piece")
            N = length(points)
            piece.attributes["NumberOfPoints"] = string(N)

            # This Points element and its associated DataArray has to be constructed individually
            points_element    = Element("Points")
            point_dataarray = create_data_array_element("Points",points,0)
            point_dataarray["offset"] = 0

            
            # Generate appended data element
            appendeddata = Element("AppendedData")
            appendeddata.attributes["encoding"] = "raw"

            # Start writing the file and generating the correct dataarrays with the right offsets in the loop
            NB = 0
            io = IOBuffer()
            write(io,"\n_")
            UncompressedHeaderN  = N * length(first(points)) *  sizeof(typeof(first(points)))
            NB += write(io, UncompressedHeaderN)
            NB += write(io, points)

            # Generate XML tags for kwargs data
            pointdata  = Element("PointData")
            dataarrays = Vector{XML.Node}(undef,length(args))

            if !isnothing(args)
                for i in eachindex(args)
                    arg           = args[i]
                    dataarrays[i] = create_data_array_element(variable_names[i],arg,NB)

                    A             = typeof(first(arg))
                    T             = eltype(A)
                    Ni            = length(arg)
                    Tsz           = sizeof(T)
                    Nc            = Int( sizeof(A) / Tsz )
                    HowManyBytes  = Tsz*Nc*Ni + Tsz

                    NB           += HowManyBytes

                    write(io, NB)
                    write(io, arg)
                end
            end

            # Take the result from the buffer, turn to string and write it
            v = take!(io)
            t = Text(String(v))
            write(io,"\n")
            push!(appendeddata,t)
            close(io)

            # Glue all xml pieces together
            push!(xml_doc,vtk_file)
            push!(points_element,point_dataarray)
            push!(piece,points_element)
            push!(polydata,piece)
            push!(vtk_file,polydata)
            map(x -> push!(pointdata,x), dataarrays)
            push!(piece,pointdata)
            push!(vtk_file,appendeddata)

            XML.write(filename,xml_doc)
    end

    function ConvertHDFtoVTP(filename::String, DictVariable)
        points = reinterpret(reshape, SVector{3,Float64}, DictVariable["Position"])
        # Generate the XML document and then put in some fixed values
        xml_doc = Document(XML.Declaration(version=1.0,encoding="utf-8"))
        vtk_file = Element("VTKFile")
        vtk_file.attributes["type"]        = "PolyData"
        vtk_file.attributes["version"]     = "1.0"
        vtk_file.attributes["byte_order"]  = "LittleEndian"
        vtk_file.attributes["header_type"] = "UInt64"
    
        # PolyData is the main section, filling it out
        polydata  = Element("PolyData")
        piece     = Element("Piece")
        N = length(points)
        piece.attributes["NumberOfPoints"] = string(N)
    
        # This Points element and its associated DataArray has to be constructed individually
        points_element    = Element("Points")
        point_dataarray = create_data_array_element("Points",points,0)
        point_dataarray["offset"] = 0
    
        
        # Generate appended data element
        appendeddata = Element("AppendedData")
        appendeddata.attributes["encoding"] = "raw"
    
        # Start writing the file and generating the correct dataarrays with the right offsets in the loop
        NB = 0
        io = IOBuffer()
        write(io,"\n_")
        UncompressedHeaderN  = N * length(first(points)) *  sizeof(typeof(first(points)))
        NB += write(io, UncompressedHeaderN)
        NB += write(io, points)
    
        # Generate XML tags for kwargs data
        pointdata  = Element("PointData")
        pop!(DictVariable,"Position")
        dataarrays = Vector{XML.Node}(undef,length(DictVariable))
    
        i = 1
        for (key,value) in DictVariable
            
            T = eltype(value)
            NumberOfFields = fieldcount(eltype(T))
    
            if NumberOfFields > 0
                T = unique(fieldtypes(eltype(T)))[1]
                data_type = SVector{NumberOfFields, T}
            else
                data_type = T
            end
    
            val = reinterpret(reshape, data_type, value)
            arg           = val
    
            dataarrays[i] = create_data_array_element(key,arg,NB)
            A             = typeof(first(arg))
            T             = eltype(A)
            Ni            = length(arg)
            Tsz           = sizeof(T)
            Nc            = Int( sizeof(A) / Tsz )
            HowManyBytes  = Tsz*Nc*Ni + Tsz
            NB           += HowManyBytes
            write(io, NB)
            write(io, arg)
    
            i += 1
        end
    
        # Take the result from the buffer, turn to string and write it
        v = take!(io)
        t = Text(String(v))
        write(io,"\n")
        push!(appendeddata,t)
        close(io)
    
        # Glue all xml pieces together
        push!(xml_doc,vtk_file)
        push!(points_element,point_dataarray)
        push!(piece,points_element)
        push!(polydata,piece)
        push!(vtk_file,polydata)
        map(x -> push!(pointdata,x), dataarrays)
        push!(piece,pointdata)
        push!(vtk_file,appendeddata)
    
        XML.write(filename,xml_doc)
    end

    function SaveHDF5!(fid::HDF5.File, group_name, variable_names, args...)
        create_group(fid, group_name)
        if !isnothing(args)
            for i in eachindex(args)
                arg           = args[i]
                var_name          = variable_names[i]
                fid[group_name][var_name] = arg
            end
        end
    end

    function HDFtoVTP(SimMetaData)
        fid = h5open(SimMetaData.SaveLocation * "/" * SimMetaData.SimulationName * ".h5","r")
        for key in keys(fid)
            DictVariable = read(fid[key])
            ConvertHDFtoVTP(SimMetaData.SaveLocation * "/" * SimMetaData.SimulationName * "_" * key * ".vtp", DictVariable)
        end
        close(fid)
    end

    function OpenForWriteH5(path)
        return h5open(path, "w")
    end
    
    # save_location = raw"E:\SPH\TestOfFile.vtp"

    # d = @report_opt target_modules=(@__MODULE__,) PolyDataTemplate(save_location, Points, ["Kernel", "KernelGradient"], Kernel, KernelGradient)
    # println(d)

    # @profview PolyDataTemplate(save_location, Points, ["Kernel", "KernelGradient"], Kernel, KernelGradient)

    # b = @benchmark PolyDataTemplate($save_location, $Points, $(["Kernel", "KernelGradient"]), $Kernel, $KernelGradient)
    # display(b)

    # @code_warntype PolyDataTemplate(save_location, Points, ["Kernel", "KernelGradient"], Kernel, KernelGradient)

    # PolyDataTemplate(save_location, Points, ["Kernel", "KernelGradient"], Kernel, KernelGradient)
    # PolyDataTemplate(save_location, Points)
end