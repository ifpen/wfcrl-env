function [yawRef] = simInterface(n_turbines, power, ws, time, com_file_yaws, com_file_power, com_file_wind)
    
    %% Initialize algorithm
    wu = ws(1,1);
    wv = ws(1,2);
    
    persistent yaw_old num_iter last_input_id
    if isempty(yaw_old)
        yaw_old = zeros(n_turbines,1);
        num_iter = int64(0);
        last_input_id = -1;
    end

% Write power info
fileID = fopen(com_file_power, 'w');
fprintf(fileID, ['%d ', repmat('%f ',1,n_turbines), '%d\n'], num_iter, power, 0);
fclose(fileID);
%     waiting_for_input = true;

% Write wind info
speed = sqrt(wu^2 + wv^2);
dir = -atan2d(wu,wv);
if dir < 0
    dir = dir + 360;
end
fileID = fopen(com_file_wind, 'w');
fprintf(fileID, '%d %f %f %d\n', num_iter, speed, dir, 0);
fclose(fileID);

while last_input_id <= num_iter
    % Read yaws
    
    fileID = fopen(com_file_yaws, 'r');
    content = fgetl(fileID);
    fclose(fileID);
    if content ~= - 1
        indices = strfind(content, " ");
        last_input_id = real(str2double(content(1:indices(1)-1)));
    end
    pause(0.05);
end

inputs = split(content, " ");
for i = 1:size(yaw_old,1)
    yaw_old(i) = real(str2double(inputs{i+1}));
end
num_iter = num_iter + 1;
   
yawRef = yaw_old;
end

