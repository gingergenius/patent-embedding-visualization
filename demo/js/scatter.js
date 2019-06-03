if(!d3.chart) d3.chart = {};

d3.chart.scatter = function() {
	var g
	var cont
	var data;
	var clusterData;
	var width = 400;
	var height = 400;
	var zoom_level = 1;
	var points_g;
	var clusters_g;
	var dispatch = d3.dispatch("hover", "endHover");
	var zoom;
	var toolTipAll;
	var toolTipAugmented;
	var bigClusterScale, mediumClusterScale, smallClusterScale;
	var lineThicknessScale, referencesScale;
	var selectedPoint;
	var xScale, yScale;
	var currentxScale, currentyScale;
	var topLayer;
	var column;
	var pointSizeMultiplierScale;
	var arc;
	var sunburstDomain;

	function chart(container) {

		cont = container;

		// create a clipping region 
		clip_path = container.append("defs").append("clipPath")
			.attr("id", "clip")
			.append("rect")
			.attr("width", width)
			.attr("height", height)

		container.on("click", resetSelectedPoint)	
			.append("rect")	
			.attr("id", "zoom")	
			.attr("width", width)
			.attr("height", height)
			.attr("stroke", "#ccc")
			.attr("stroke-width", 2)
			.attr("fill", "none")
			.style("pointer-events", "all")	

		// Draw Datapoints
		g = container.append("g")
			.attr("clip-path", "url(#clip)")

		points_g = g.append("g")
			.classed("points", true)

		clusters_g = g.append("g")
			.classed("clusters", true)

		/* Initialize tooltip */
		toolTipAugmented = d3.tip()
			.attr('class', 'd3-tip-augmented')
			.offset([0, 0])
			.direction("s")
			.html(function(l) { 
				return l.join(", ")
			});
		/* Invoke the tip in the context of your visualization */
		clusters_g.call(toolTipAugmented);

		toolTipAll = d3.tip()
		.attr('class', 'd3-tip-all')
		.offset([0, 0])
		.direction("n")
		.html(function(l) { 
			return l.join(", ")
		});

		clusters_g.call(toolTipAll);

		// Pan and zoom
		zoom = d3.zoom()
			.scaleExtent([.5, 10])
			.extent([[0, 0], [width, height]])
			.on("zoom", zoomed);

		cont.select("#zoom").call(zoom);

		bigClusterScale = d3.scaleLinear()
			.domain([0.8, 1.6])
			.range([23, 28])
			.clamp(true);

		mediumClusterScale = d3.scaleLinear()
			.domain([1.5, 2.1])
			.range([18, 22])
	
		smallClusterScale = d3.scaleLinear()
			.domain([2.0, 3])
			.range([14, 18])
			.clamp(true);

		lineThicknessScale = d3.scaleLinear()
			.domain([0.7, 3])
			.range([2, 3])
			.clamp(true)

		xScale = d3.scaleLinear();
		yScale = d3.scaleLinear();

		pointSizeMultiplierScale = d3.scaleLinear()
			.domain([0.7, 3])
			.range([1, 1.7])
			.clamp(true)

		arc = d3.arc()           
			.innerRadius(0)
	}
	chart.update = update;

	function update() {
		
		referencesScale = d3.scaleLinear()
			.domain(d3.extent(data, function(d) { return d.references.length + d.referenced_by.length }))
			.range([3, 9])

		var extentX = d3.extent(data, function(d) { return d.x})
		var extentY = d3.extent(data, function(d) { return d.y})

		xScale.domain([extentX[0] - 2, extentX[1] + 2]) //not only points have to fit into viewports, but also corresponding circles
			.range([0, width])
		
		yScale.domain([extentY[0] - 2, extentY[1] + 2])
			.range([height, 0])

		currentxScale = xScale;
		currentyScale = yScale;

		chart.resetZoom();

		drawNodes();
		drawClusters();
		selectedPoint = null;		
	}

	function drawNodes() {

		var activePoints = points_g.selectAll("g.node").select(function() { 
			return d3.select(this).classed("inactive_sunburst") || d3.select(this).classed("inactive_histogram") ? null : this })
		
		var nodesInView = activePoints.selectAll("circle").filter(function() { 
			cx = this.getAttribute("cx");
			cy = this.getAttribute("cy");
			return (cx > 0) && (cx < width) && (cy > 0) && (cy < height)
		}).size();

		if (nodesInView == 0) {
			nodesInView = data.length; //on the first run no circles are there yet. Workaround.
		}
		//console.log("Zoom level:", zoom_level);

		//show fewer labels if zoomed out far
		var everyNth = nodesInView / 250

		// DATA JOIN
		// Join new data with old elements, if any.
		var nodesData = points_g.selectAll("g.node")
			.data(data, function(d) { return d.x })

		//ENTER 
		var nodesEnter = nodesData.enter()
			.append("g")
			.classed("node", true)
			
		nodesEnter.append("text")

		nodesEnter.append("g")
			.classed("glyph", true)

		nodesEnter.append("circle")
			.style("fill-opacity", "0")
			.on("click", clickOnPoint)
			.on("mouseenter", drawConnections)
			.on("mouseleave", hideConnections)
			
		//UPDATE	
		nodesData.merge(nodesEnter).selectAll("circle")
			.attrs({
				cx: function(d) { return currentxScale(d.x) },
				cy: function(d) { return currentyScale(d.y) },
				r: function(d) { return referencesScale(d.references.length + d.referenced_by.length) * pointSizeMultiplierScale(zoom_level)}
			})
		nodesData.merge(nodesEnter).selectAll("text")
			.text(getZoomedTerms)
			.attrs({
				class: "node_label",
				x: function(d) { return currentxScale(d.x) + 
					(referencesScale(d.references.length + d.referenced_by.length) / 2) + 5 },
				y: function(d) { return currentyScale(d.y) + 
					(referencesScale(d.references.length + d.referenced_by.length) / 2) }	
			})
		nodesData.merge(nodesEnter).select("g.glyph").selectAll("path")
			.attr("transform", function() {
				var circle = d3.select(this.parentNode.parentNode).select("circle")
				return "translate(" + [circle.attr("cx"), circle.attr("cy")] + ")" 
			})
			.attr("d", function(d) {
				var circle = d3.select(this.parentNode.parentNode).select("circle")
				arc.outerRadius(circle.attr("r"))
				return arc(d)
			})

		//EXIT
		nodesData.exit().remove();

		function getZoomedTerms(d){
			var num_terms = 0
			if (zoom_level > 1.15){
				if (everyNth > 1) {
					if (+d.pub_num % Math.ceil(everyNth) > 0) {
						num_terms = 0
					} else {
						num_terms = 1
					}
				} else if (everyNth > 0.7) {
					num_terms = 1
				} else if (everyNth > 0.3) {
					num_terms = 2
				} else {
					num_terms = 3
				}
			}
			return d.terms.slice(0, num_terms).join(", ");
		}
	}

	function drawClusters() {
		
		// remove old clusters before drawing new ones
		clusters_g.selectAll("text").remove();

		if (clusterData) {
			// DATA JOIN
			// Join new data with old elements, if any.
			var clust_data = clusters_g.selectAll("text")
			.data(clusterData, function(d) {return d.id})
			
			// ENTER
			clust_enter = clust_data.enter()
				.append("text")
					.attr("class", function(d) { return d.level})
					.classed("cluster_title", true)
					.attr("text-anchor", "middle")
					.each(function(d) {
						
						for (t in d.labels.slice(0,3)) {
							// Add white contour
							d3.select(this).append('tspan')
								.attr("x", 0)
								.attr("dy",  1.2 * t * Math.pow(-1, t%2)  +"em")
								.text(d.labels[t])
								.style('fill', 'none')
								.style('stroke', '#fff')
								.style('stroke-width', 3)
								.style('stroke-linejoin', 'round')
								.classed("outline", true)

							d3.select(this).append("tspan")
								.text(d.labels[t])
								.attr("x", 0)
								.attr("dy", 0)		
								.classed("cluster_term", true)
								.on("mouseenter", showAugmentedTerms)	
								.on("mouseleave", hideTooltip)													

							d3.select(this).select(".outline").classed("first_outline", true);
							d3.select(this).select(".cluster_term").classed("first_cluster_term", true);
						}						
					})

			// ENTER + UPDATE
			// After merging the entered elements with the update selection,
			// apply operations to both.	
			clust_data.merge(clust_enter)
				.attr("transform", function(d) {
					return "translate (" + currentxScale(d.x) + " " + currentyScale(d.y) + ")"
				})
				.style("font-size", function(d) {
					var s = 20;
					switch(d.level) {
						case "big":
							s = bigClusterScale(zoom_level)
							break;
						case "medium":
							s = mediumClusterScale(zoom_level)
							break;
						case "small":
							s = smallClusterScale(zoom_level)
							break;
					}
					return s + "px"
				})
				.style("visibility", function(d) {
					switch(d.level) {
						case "big":
							fill = (zoom_level <= 1.6) ? "visible" : "hidden"
						break;
						case "medium":
							fill = (zoom_level > 1.5 && zoom_level <= 2.1) ? "visible" : "hidden"
						break;
						case "small":
							fill = (zoom_level > 2.0 && zoom_level < 3.5) ? "visible" : "hidden"
						break;
						default:
							fill = "visible"
					}
					return fill
				})
				.select(".first_outline")
				.style("font-size", enlargeFirstTerm)

				clust_data.merge(clust_enter)
				.select(".first_cluster_term")
				.style("font-size", enlargeFirstTerm)

				function enlargeFirstTerm(d) {
					var s = 20;
					switch(d.level) {
						case "big":
							s = bigClusterScale(zoom_level)
							break;
						case "medium":
							s = mediumClusterScale(zoom_level)
							break;
						case "small":
							s = smallClusterScale(zoom_level)
							break;
					}
					return s + 3 + "px" //first term bigger
				}
			
			// EXIT
			// Remove old elements as needed.
			clust_data.exit().remove();
		}
		
	}

	function hideTooltip() {
		toolTipAugmented.hide()
		toolTipAll.hide()
	}
	
	function showAugmentedTerms(d) {
		var label = d3.select(this).text()
		if (label in d.augmented_labels){
			terms = d.augmented_labels[label]
			if (terms.length > 0) {
				toolTipAugmented.show(terms);
			}				
		}			
		toolTipAll.show(d.labels.slice(3, 15))	
	}

	function redrawLines() {
		//redrawing family and citations on new zoom level
		points_g.selectAll("line")
			.attr("x1", function() { return currentxScale(this.getAttribute("xx1")) })     
			.attr("y1", function() { return currentyScale(this.getAttribute("yy1")) })  
			.attr("x2", function() { return currentxScale(this.getAttribute("xx2")) })  
			.attr("y2", function() { return currentyScale(this.getAttribute("yy2")) })  
			.attr("stroke-width", function () { return lineThicknessScale(zoom_level)})		
	}

	function zoomed() {
		// create new scale objects based on event
		currentxScale = d3.event.transform.rescaleX(xScale);
		currentyScale = d3.event.transform.rescaleY(yScale);

		zoom_level = d3.event.transform.k;

		drawNodes();	
		drawClusters();
		redrawLines();
	}

	function resetSelectedPoint() {
		//console.log("reset selected point")
		if (selectedPoint) {
			points_g.select("g." + selectedPoint.publication_number).select("circle")
				.on("mouseenter", drawConnections)
				.on("mouseleave", hideConnections);					
			hideConnections(selectedPoint, null, null, remove_persistent = true);				
			selectedPoint = null;				
		}	
	}

	function clickOnPoint(d) {
		circle = points_g.selectAll("g.node").filter(function(e) {return e.publication_number == d.publication_number}).select("circle")

		if (selectedPoint == d) { 
			selectedPoint = null;						
			hideConnections(d, null, null, remove_persistent=true)

			circle.on("mouseenter", drawConnections)
			circle.on("mouseleave", hideConnections)					
		} else {
			resetSelectedPoint();
			selectedPoint = d;
			drawConnections(d, null, null, persistent=true)						

			circle.on("mouseenter", null)
			circle.on("mouseleave", null)
		}		
		d3.event.stopPropagation();		
	}

	function hideConnections(d, i, t, remove_persistent=false) {
		nodes = points_g.selectAll("g." + d.publication_number)
			.classed(d.publication_number, false);
		
		filteredCircles = nodes.select(function () {
			cl = d3.select(this).attr("class");
			return cl.includes(" ") ? null : this //g has not only class "node" but a leftover class from currently clicked point
		})
		.selectAll("circle")
			.style("stroke", "");
	
		lines = nodes.selectAll("line");
		if (remove_persistent) {
			filteredLines = lines		
		} else {
			filteredLines = lines.select(function () { return d3.select(this).classed("persistent") ? null : this })
		}		
		filteredLines.remove();
		
		dispatch.call("endHover", this, d);
	}

	function drawConnections(d, i, t, persistent=false) {

		circle = points_g.selectAll("g.node").filter(function(e) {return e.publication_number == d.publication_number}).select("circle")

		parent_g = circle.select(function() { return this.parentNode; })

		if (!(parent_g.classed("inactive_histogram") || parent_g.classed("inactive_sunburst"))) {
			circle.style("stroke", "black")
				.attr("stroke-width", function () { return lineThicknessScale(zoom_level)})

			parent_g.classed(d.publication_number, true)

			circles = points_g.selectAll("g.node").select(function(e) { 
				var inactive = d3.select(this).classed("inactive_histogram") || d3.select(this).classed("inactive_sunburst")
				return inactive ? null : this;
			}).select("circle");
			
			var family = circles.select(function(e) { 
				return (e.family_id == d.family_id && d.publication_number != e.publication_number) ?  this : null; //exclude node itself from family
			});
			family.each(function (e) { drawLines (this, d, e, persistent) })

			var cited_by = circles.select(function(e) { 
				return d.referenced_by.includes(e.publication_number) ? 
					this : null;
			});
			cited_by.each(function (e) { drawLines (this, d, e, persistent, dashed=true, color="#009BFF") })	

			var citing = circles.select(function(e) { 
				return d.references.includes(e.publication_number) ? 
					this : null;
			});
			citing.each(function (e) { drawLines (this, d, e, persistent, dashed=true, color="#FFB20A")})	

			citing.merge(cited_by).merge(family)
				.attr("stroke-width", function () { return lineThicknessScale(zoom_level)})
				.raise()

			// //bring all circle wrappers to foreground
			//family.parentNode.raise();
			//referenced.parentNode.raise();				
							
			parent_g.raise();
							
			dispatch.call("hover", this, d);		
		}					
	}

	function drawLines(node, d, e, persistent, dashed = false, color="black") {
		d3.select(node).style("stroke", color)
		d3.select(node.parentNode)
			.classed(d.publication_number, true)   
			.raise()

		line = d3.select(node.parentNode).insert("line", "circle")    
			.style("stroke", color)
			.attr("stroke-width", function () { return lineThicknessScale(zoom_level)})
			.attr("xx1", e.x)     
			.attr("yy1", e.y)     
			.attr("xx2", d.x)     
			.attr("yy2", d.y)   			
			.attr("x1", currentxScale(e.x))     
			.attr("y1", currentyScale(e.y))     
			.attr("x2", currentxScale(d.x))     
			.attr("y2", currentyScale(d.y))   	
			.classed("persistent", function(d) {
				alreadyPersistent = d3.select(this).classed("persistent");
				return alreadyPersistent? true: persistent;
			 }) 
		if (dashed) {
			line.style("stroke-dasharray", [5,5]) 
		}
	}	

	chart.filter = function(filtered) {
		points = points_g.selectAll("g.node")
			.classed("inactive_histogram", true)
		
		points.data(filtered, function(d) { return d.publication_number })
			.classed("inactive_histogram", false)

		drawNodes();
	}

	chart.highlight = function(hovered) {				
		points = points_g.selectAll("g.node");
		points.classed("inactive_sunburst", function(d) { 
			return (hovered.data.data.ids.includes(d.publication_number)) ? false : true; 
		})
		drawNodes();			
	}

	chart.matchColors = function(newTopLayer, newDomain) {

		var pie = d3.pie()           
			.value(function(d) { return 1}); 
		
		if (newTopLayer) {
			topLayer = newTopLayer
		}
		if (newDomain) {
			sunburstDomain = newDomain
		}

		const x = d3.scaleLinear()
            .range([0, 2 * Math.PI])
			.clamp(true)
			.domain(sunburstDomain);

        function getColor(d) {
            var mid_angle = x((d.x0 + d.x1) / 2)
            var mid = mid_angle / (Math.PI * 2)

			const epsilon = 0.0001

            if (mid < 0.5 + epsilon && mid > 0.5 - epsilon) {                
				var color = d3.rgb(115, 127, 150)
			} else {
				var color = d3.interpolateRainbow(mid)                                                       
			}

            return color;
        }

		var children = topLayer.children;
		if (children) {

			column = children[0].data.data.column;

			var colors = {} 
			
			children.map(function (d) {
				var name = d.data.data.name;
				var color = getColor(d);
				colors[name] = color
			})

			//console.log(colors)

			points_g.selectAll("g.node").each(function(d) {
				var node = d3.select(this)
				var circle = node.select("circle");

				// Remove old elements as needed.
				arc.outerRadius(circle.attr("r"))

				node.select("g.glyph").selectAll("path").remove();

				var allowedValues = Object.keys(colors);
				var pointAllowedValues = [];

				if (column == "country_code") {
					if (allowedValues.includes(d[column])) {
						pointAllowedValues = [d[column]]
					}					
				} else {
					pointAllowedValues = d[column].filter(function(e) { return allowedValues.includes(e) ? e : null; } )
				}

				if (pointAllowedValues.length > 0) {
					//JOIN
					var slicesData = node.select("g.glyph").selectAll("path").data(pie(pointAllowedValues))

					//ENTER
					var slicesDataEnter = slicesData.enter()
						.append("path")

					//ENTER + UPDATE
					var slicesMerged = slicesData.merge(slicesDataEnter)
							.attr("d", arc)
							.attr("transform", "translate(" + [circle.attr("cx"), circle.attr("cy")] + ")" )
							.attr("fill", function(e) { return colors[e.data] } )
				}				
			})
		}
	}

	chart.data = function(value) {
		if(!arguments.length) return data;
		data = value;
		update();
		return chart;
	}
	chart.clusterData = function(value) {
		if(!arguments.length) return clusterData;
		clusterData = value;
		drawClusters();
		return chart;
	}
	chart.width = function(value) {
		if(!arguments.length) return width;
		width = value;
		return chart;
	}
	chart.height = function(value) {
		if(!arguments.length) return height;
		height = value;
		return chart;
	}

	chart.resetZoom = function(){
		cont.select("#zoom").call(zoom.transform, d3.zoomIdentity);
		zoom_level = 1;
	}

	chart.getSelectedPoint = function() {
		return selectedPoint;
	}

	return d3.rebind(chart, dispatch, "on");
}