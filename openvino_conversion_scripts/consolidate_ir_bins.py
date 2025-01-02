from bs4 import BeautifulSoup
import os
import sys
from pathlib import Path
import numpy as np


class ConstDatabaseEntry:
    def __init__(self, ir_basename, const_vals, name):
       self.const_vals = const_vals #the actual tensor vals

       self.const_names_dict = {}
       self.const_names_dict[ir_basename] = [name] # a list of const names that use this tensor

       self.offset = 0 # to be filled later on, as we're writing stuff.

    def compare(self, const_vals):
        return np.array_equal(self.const_vals, const_vals)

    def add(self, ir_basename, const_name):

        ir_const_list = self.const_names_dict.get(ir_basename)
        if ir_const_list is None:
            self.const_names_dict[ir_basename] = []

        if not const_name in self.const_names_dict[ir_basename]:
            self.const_names_dict[ir_basename].append(const_name)

            #print("self.const_names_list[", ir_basename, "] = ", self.const_names_dict[ir_basename])


class ConstDatabase:
    def __init__(self):
       self.entry_dict = {} # dictionary of key=size -> list of ConstDatabaseEntry's

       self.ir_to_const_entry_dict = {} # name of IR -> another dict { name of const -> ConstDatabaseEntry }
       self.entrylist = []
       self.total_entries_size = 0
       self.ir_to_xml = {}

    def add_to_database(self, ir_basename, constname, const_vals):
        #print("add_to_database called!")
        const_vals_size = const_vals.shape[0]
        #print("    const_vals_size = ", const_vals_size)
        entry_list = self.entry_dict.get(const_vals_size)
        entry = None
        if entry_list is None:
            #print("creating new entry for ", const_vals_size)
            entry = ConstDatabaseEntry(ir_basename, const_vals, constname)
            self.entry_dict[const_vals_size] = [entry]
            self.entrylist.append(entry)
            self.total_entries_size += const_vals_size
        else:
            found_match = False
            for e in entry_list:
                if e.compare(const_vals):
                    found_match = True
                    entry = e
                    entry.add(ir_basename, constname)
                    break

            #okay, none of the const arrays of this size match.. so just add a new entry.
            if found_match==False:
                #print("appending new entry for size = ", const_vals_size)
                entry = ConstDatabaseEntry(ir_basename, const_vals, constname)
                entry_list.append(entry)
                self.entrylist.append(entry)
                self.total_entries_size += const_vals_size

            #print("found existing entry list for ", const_vals_size)
        if not ir_basename in self.ir_to_const_entry_dict:
            self.ir_to_const_entry_dict[ir_basename] = {}

        self.ir_to_const_entry_dict[ir_basename][constname] = entry

    # Get the size of the consts that apply *only* to this IR.
    def get_size_of_specific_ir(self, input_ir_filename):
        ir_basename = os.path.basename(input_ir_filename)
        size = 0
        for entry in self.entrylist:
            if len(entry.const_names_dict) == 1 and ir_basename in entry.const_names_dict:
                size += entry.const_vals.shape[0]

        return size


def load_ir_to_database(const_database, input_ir_filename):
    binfile = os.path.join(os.path.dirname(input_ir_filename), Path(input_ir_filename).stem + ".bin")
    binstats = os.stat(binfile)

    ir_basename = os.path.basename(input_ir_filename)
    print("ir_basename = ", ir_basename)
    print("binfile = ", binfile)
    print("binfile size = ", binstats.st_size)
    binarray = np.fromfile(binfile, dtype='int8')

    #print("binarray shape = ", binarray.shape)

    print("input_ir_filename = ", input_ir_filename)
    with open(input_ir_filename, 'r') as f:
        data = f.read()
        
    print("done reading ", input_ir_filename)

    ir_xml = BeautifulSoup(data, "xml")

    const_database.ir_to_xml[ir_basename] = ir_xml

    layers = ir_xml.find_all('layer')

    # display content
    for layer in layers:
        if( layer.get('type') == 'Const'):
            name = layer.get('name')
            #print("name = ", name)
            data = layer.findChildren('data')
            #print("type(data) = ", type(data))
            for d in data:
               #print(print(d))
               offset =  int(d.get('offset'))
               size = int(d.get('size'))
               #print("   offset = ", offset)
               #print("   size = ", size)
               constvals = binarray[offset:offset+size]
               const_database.add_to_database(ir_basename, name, constvals)

    return ir_xml

def consolidate_ir_bins_in_folder(input_folder=None, input_ir_files=None, output_folder=None, combined_bin_name=None):
    if input_ir_files is None:
        if input_folder is None:
            print("Error! Either input_ir_files or input_folder parameter must be set")
            sys.exit(1)
            
        input_ir_files = []
        for filename in os.listdir(input_folder):
            f = os.path.join(input_folder, filename)
            if os.path.isfile(f):
                if Path(f).suffix == ".xml":
                    input_ir_files.append(f)

    if output_folder is not None:
        if not os.path.isdir(output_folder):
            print(f"output_folder, {output_folder},  does not exist!")
            sys.exit(1)

    print("IR's that will be processed: ",   input_ir_files)

    cdb = ConstDatabase()

    #1. Take each IR, and load their const's to the database.
    for ir_file in input_ir_files:
        print("Processing", ir_file, "...")
        load_ir_to_database(cdb, ir_file)

    print("total const entry size = ", cdb.total_entries_size)

    #2. Pre-allocate a numpy array that is the total size of the database.
    # We'll slice values in as we go... this is (probably?) much faster
    # than expanding the array over and over again with each new value.
    current_bin_index=0
    combined_bin_array = np.zeros(cdb.total_entries_size, dtype='int8')

    print("total size of combined bin files = ", cdb.total_entries_size)

    for ir_file in input_ir_files:
        ir_specific_size = cdb.get_size_of_specific_ir(ir_file)
        print("Total size in const file that is only used by", os.path.basename(ir_file), "=", ir_specific_size)


    #3. Iterate through the database's entrylist, and slice these entries into
    #   combined_bin_array. We also set each entry offset here, which we eventually
    #   use to modify the xml
    for entry in cdb.entrylist:
        combined_bin_array[current_bin_index:current_bin_index + entry.const_vals.shape[0]] = entry.const_vals

        #set the entry's offset.. this will be used later on when we update the xml files
        entry.offset = current_bin_index

        current_bin_index += entry.const_vals.shape[0]



    for ir_basename, ir_xml in cdb.ir_to_xml.items():
        layers = ir_xml.find_all('layer')

        # display content
        for layer in layers:
            if( layer.get('type') == 'Const'):
                name = layer.get('name')
                #print("name = ", name)
                data = layer.findChildren('data')
                #print("type(data) = ", type(data))
                for d in data:
                   #print(print(d))
                   #get the entry for this const value
                   entry = cdb.ir_to_const_entry_dict[ir_basename][name]

                   #update the offset in the xml with the new one
                   d['offset'] = entry.offset

        if output_folder is not None:
            new_xml_file = Path(ir_basename).stem + ".xml"
            new_xml_file = os.path.join(output_folder, new_xml_file)
            with open(new_xml_file, 'w') as file:
                file.write(str(ir_xml.prettify()))
                file.close()

    if output_folder is not None:
        # write the combined bin file to disk
        if combined_bin_name is None:
            combined_bin_name = "combined.bin"
        combined_bin_array.tofile(os.path.join(output_folder, combined_bin_name))

    print("done!")


if __name__ == "__main__":
    consolidate_ir_bins_in_folder(input_folder='input_folder', output_folder='output_folder')


