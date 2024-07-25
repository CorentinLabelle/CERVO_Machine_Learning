function generate_mat_config_from_ini(ini_path)

    [ini_folder, ini_filename, ~] = fileparts(ini_path);
    mat_config_path = fullfile(ini_folder, [ini_filename '.mat']);
    
    ini_struct = ini2struct(ini_path);
    save(mat_config_path, "-struct", "ini_struct");
    