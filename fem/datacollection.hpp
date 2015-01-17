// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_DATACOLLECTION
#define MFEM_DATACOLLECTION

#include "../config.hpp"
#include <string>
#include <map>

namespace mfem
{

/** A class for collecting finite element data that is part of the same
    simulation. Currently, this class groups together several grid functions
    (fields) and the mesh that they are defined on. */
class DataCollection
{
protected:
   /// Name of the collection, used as a directory name when saving
   std::string name;

   /// The fields and their names (used when saving)
   std::map<std::string, GridFunction*> field_map;
   /// The (common) mesh for the collected fields
   Mesh *mesh;

   /// Time cycle (for time-dependent simulations)
   int cycle;
   /// Physical time (for time-dependent simulations)
   double time;

   /// Serial or parallel run?
   bool serial;
   /// MPI rank (in parallel)
   int myid;
   /// Number of MPI ranks (in parallel)
   int num_procs;

   /// Should the collection delete its mesh and fields
   bool own_data;

public:
   /// Create an empty collection with the given name.
   DataCollection(const char *collection_name);
   /// Initialize the collection with its mesh
   DataCollection(const char *collection_name, Mesh *_mesh);

   /// Add a grid function to the collection
   virtual void RegisterField(const char *field_name, GridFunction *gf);
   /// Get a pointer to a grid function in the collection
   GridFunction *GetField(const char *field_name);
   /// Check if a grid function is part of the collection
   bool HasField(const char *name) { return field_map.count(name) == 1; }
   /// Get a pointer to the mesh in the collection
   Mesh *GetMesh() { return mesh; }

   /// Set time cycle (for time-dependent simulations)
   void SetCycle(int c) { cycle = c; }
   /// Set physical time (for time-dependent simulations)
   void SetTime(double t) { time = t; }

   /// Get time cycle (for time-dependent simulations)
   int GetCycle() { return cycle; }
   /// Get physical time (for time-dependent simulations)
   double GetTime() { return time; }
   /// Get the name of the collection
   const char* GetCollectionName() { return name.c_str(); }
   /// Set the ownership of collection data
   void SetOwnData(bool o) { own_data = o; }

   /** Save the collection to disk. By default, everything is saved in a
       directory with name <collection_name> or <collection_name>_cycle for
       time-dependent simulations. */
   virtual void Save();

   /// Delete the mesh and fields if owned by the collection
   virtual ~DataCollection();
};


/// Helper class for VisIt visualization data
class VisItFieldInfo
{
public:
   std::string association;
   int num_components;
   VisItFieldInfo() { association = ""; num_components = 0; }
   VisItFieldInfo(std::string _association, int _num_components)
   { association = _association; num_components = _num_components; }
};

/// Data collection with VisIt I/O routines
class VisItDataCollection : public DataCollection
{
protected:
   // Additional data needed in the VisIt root file, which describes the mesh
   // and all the fields in the collection
   int spatial_dim, topo_dim;
   int visit_max_levels_of_detail;
   int num_ranks;
   std::map<std::string, VisItFieldInfo> field_info_map;

   /// Prepare the VisIt root file in JSON format for the current collection
   std::string GetVisItRootString();
   /// Read in a VisIt root file in JSON format
   void ParseVisItRootString(std::string json);

   // Helper functions for LoadVisItData()
   void LoadVisItRootFile(std::string root_name);
   void LoadMesh();
   void LoadFields();

public:
   /** Create an empty collection with the given name, that will be filled in
       later with the Load() function. Currently this only works in serial! */
   VisItDataCollection(const char *collection_name);
   /// Initialize the collection with its mesh, fill-in the extra VisIt data
   VisItDataCollection(const char *collection_name, Mesh *_mesh);

   /// Add a grid function to the collection and update the root file
   virtual void RegisterField(const char *field_name, GridFunction *gf);

   /// Set additional VisIt parameters
   void SetVisItParameters(int max_levels_of_detail);

   /// Save the collection and a VisIt root file
   virtual void Save();

   /// Load the collection based on its VisIt data (described in its root file)
   void Load(int _cycle = 0);

   /// We will delete the mesh and fields if we own them
   virtual ~VisItDataCollection() {}
};

}

#endif
