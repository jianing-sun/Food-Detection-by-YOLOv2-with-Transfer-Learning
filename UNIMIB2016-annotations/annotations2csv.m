% This code used to save annotations of food dataset UNIMIB2016 to a csv
% file (data_js.csv). Each image has single or multiple objects with 6 cells
% which has been explained below.
% The converted csv file used to be analyzed for object detection.
% Jianing Sun, May 13th.

function demo()
path_to_images = './original/';

% Load the annotations in a map structure
load('annotations.mat');

% Each entry in the map corresponds to the annotations of an image.
% Each entry contains many cell tuples as annotated food
% A tuple is composed of 8 cells with the annotated:
% - (1) item category (food for all tuples)
% - (2) item class (e.g. pasta, patate, ...)
% - (3) item name
% - (4) boundary type (polygonal for all tuples)
% - (5) item's boundary points [x1,y1,x2,y2,...,xn,yn]
% - (6) item's bounding box [x1,y1,x2,y2,x3,y3,x4,y4]

image_names = annotations.keys;

% test
tuple = annotations('20151127_115133');
[nr, nc] = size(tuple);
name(1:nr, 1) = {'20151127_115133'};
tuple = [name, tuple];

%% Write file
datei = fopen('data_js.csv', 'w');
separator = ';';
decimal = '.';
excelYear = 2013;
n_images = numel(image_names);

for j=1:n_images
    image_name = image_names{j};
    
    cellArray = annotations(image_name);
    [nr, nc] = size(cellArray);
    name = cell(nr, 1);
    name(1:nr, 1) = {image_name};
    cellArray = [name, cellArray];
    
    for z=1:size(cellArray, 1)
        %     size(cellArray, 1)
        for s=1:size(cellArray, 2)
            %         size(cellArray, 2)
            var = eval(['cellArray{z,s}']);
            % If zero, then empty cell
            if size(var, 1) == 0
                var = '';
            end
            % If numeric -> String
            if isnumeric(var)
                var = num2str(var);
                % Conversion of decimal separator (4 Europe & South America)
                % http://commons.wikimedia.org/wiki/File:DecimalSeparator.svg
                if decimal ~= '.'
                    var = strrep(var, '.', decimal);
                end
            end
            % If logical -> 'true' or 'false'
            if islogical(var)
                if var == 1
                    var = 'TRUE';
                else
                    var = 'FALSE';
                end
            end
            % If newer version of Excel -> Quotes 4 Strings
            if excelYear > 2000
                var = ['"' var '"'];
            end
            
            % OUTPUT value
            fprintf(datei, '%s', var); % Write formatted data to text file.
            
            % OUTPUT separator
            if s ~= size(cellArray, 2)
                fprintf(datei, separator);
            end
        end
        
        if z ~= size(cellArray, 1) % prevent a empty line at EOF
            % OUTPUT newline
            fprintf(datei, '\n');
        end
    end
end

% Closing file
fclose(datei);
% END

end
% END



