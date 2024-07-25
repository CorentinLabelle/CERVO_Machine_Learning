function main_training(training_config_filepath)

    command_no_argument = 'python3.9 main_training.py';

    command = [...
        command_no_argument ' ' ...
        '--training_input_file ' char(training_config_filepath)];

    system(command);
    
end