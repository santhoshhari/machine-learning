if(!require("tidyverse")){install.packages("tidyverse", repos = "http://cran.us.r-project.org")}
if(!require("magrittr")){install.packages("magrittr", repos = "http://cran.us.r-project.org")}
if(!require("fields")){install.packages("fields", repos = "http://cran.us.r-project.org")}
if(!require("plotly")){install.packages("plotly", repos = "http://cran.us.r-project.org")}
if(!require("ggplot2")){install.packages("ggplot2", repos = "http://cran.us.r-project.org")}


myKMeans <- function(nReps = 10, myScatterInput, myClusterNum = 2, maxIter = 10000, graph = T){
  
  # Initialize data structures for later usage
  clust_list <- vector("list",nReps)
  dist_list = vector("list", nReps)
  tempDF <- as.matrix(myScatterInput)
  
  # Loop through the clustering algorithm nReps times
  for(i in 1:nReps){
    # Initialize two cluster vectors of size equal to the number of data points
    # clust stores cluster ids of previous iteration
    # clust has all 1s for first iteration
    # clust_dup stores cluster ids calculated in current iteration
    # clust_dup - random assignment for first iteration
    clust <- rep(1,nrow(myScatterInput))
    clust_dup <- sample(1:myClusterNum, nrow(myScatterInput), replace = T)
    # Initialize step counter to 0
    step <- 0
    
    # Execute clustering loop until one of the following two conditions are met
    # 1. Maximum limit for number of iterations reached
    # 2. Subsequent cluster assignments are unchanged
    while( step <= maxIter & !isTRUE(all.equal(clust,clust_dup))){
      clust <- clust_dup
      # Calculate cluster centroids
      mean_clust <- rowsum(tempDF, clust)/as.numeric(table(clust))
      # Calculate distance between each data point and cluster centroid
      dist_means <- rdist(tempDF, mean_clust)
      # Update cluster ids based on the minimum distance from centroids
      clust_dup <- vapply(1:nrow(tempDF),
                          function(x) which.min(dist_means[x,]),integer(1))
      # Increment step counter
      step <- step+1
    }
    # Store the sum of Euclidean distances from centroids 
    dist_list[[i]] <- sum(vapply(1:nrow(tempDF),
                                 function(x) dist_means[x,clust_dup[[x]]],double(1)))
    # Store the cluster ids in a list
    clust_list[[i]] <- clust
  }
  min_clustid <- which.min(dist_list)
  if(graph == T){
    # Print the lowest sum of Euclidean distances to console
    print(dist_list[[min_clustid]])
    if(ncol(myScatterInput)==2){
      
      # Combine input data frame with cluster and 2D plot
      fig_2d <- cbind(myScatterInput, "clust" = clust_list[[min_clustid]]) %>%
        mutate(clust = as.factor(clust)) %>% 
        ggplot()+
        geom_point(aes(x=myScatterInput[1], y=myScatterInput[2], color = clust))+
        labs(x = "\nFeature 1", y = "Feature 2\n", title = "Data clustered on Features 1 and 2")+
        theme(plot.title = element_text(hjust = 0.5))+
        scale_color_discrete(name="Cluster")
      
      print(fig_2d)
    }
    else if(ncol(myScatterInput)==3){
      
      # Combine input data frame with cluster and 3D plot
      plot_data <- cbind(myDF, "clust" = clust_list[[min_clustid]])
      plot_data$clust <- as.factor(plot_data$clust)
      fig_3d <- plot_ly(x=plot_3d[[1]], y=plot_3d[[2]], z=plot_3d[[3]],
                        type = "scatter3d", mode = "markers",
                        color = plot_3d[["clust"]], marker = list(size = 2)) %>%
        add_markers() %>%
        layout(scene = list(xaxis = list(title = 'Feature 1'),
                            yaxis = list(title = 'Feature 2'),
                            zaxis = list(title = 'Feature 3')))
      
      print(fig_3d)
    }
  }
}
