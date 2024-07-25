function main_sequenceness(sequenceness_config_filepath)

    command_no_argument = 'python3.9 main_sequenceness.py';

    command = [...
        command_no_argument ' ' ...
        '--sequenceness_input_file ' char(sequenceness_config_filepath)];
    
    system(command);

end