function demo()
  path_to_images = 'original/';

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
  
  colors = {'red','green','yellow','black','magenta','green','blue'};

  image_names = xx.keys;

  n_images = numel(image_names);

  for j = 1 : n_images
  
    image_name = image_names{j}; 
    
    im = imread([path_to_images image_name '.jpg']);

    tuples = annotations(image_name);
        
    count = size(tuples,1);

    % BoundingBox
    annotated = insertShape(im,'Polygon', tuples(:,6),'Color','White','Linewidth',5);
    
    % Region
    annotated = insertShape(annotated,'FilledPolygon', tuples(:,5),'Color',colors(1:count),'Opacity',0.3);
  
    xy=cellfun(@textXYcoords,tuples(:,5), 'UniformOutput', false);
    
    annotated = insertText(annotated,cell2mat(xy),tuples(:,2),'FontSize',50);

    imshow(annotated);
    
    waitforbuttonpress
  end
  
end

function out = textXYcoords(arr)
  ax = mean(arr(1:2:end));
  ay = mean(arr(2:2:end));  
  out=[ax,ay];  
end
