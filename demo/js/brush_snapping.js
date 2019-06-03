if(!d3.chart) d3.chart = {};

d3.chart.snapping_histogram = function() {

	var g;
	var data;
	var filteredData;
	var height = 100;
	var width = 500;
	var dispatch = d3.dispatch("filter");
	var brush_years;
	var x_g, y_g;
	var brushg;
	var rects_g;
	var rectsOverlay_g;
	var brushYearStart, brushYearEnd;
	var thresholds;
	var xScale, yScale;
	var filteredDepth = 0;

	function chart(container) {
		g = container;

		rects_g = g.append("g")
			.classed("rects", true)

		rectsOverlay_g = g.append("g")
			.classed("rects_overlay", true)

		brush_years = g.append("text")
				.attr("id", "brushYears")
				.attr("x", 35)
				.attr("y", 12);

		// x axis
		x_g = g.append("g")
			.classed("x axis", true)
			.attr("transform", "translate(0," + height + ")")

		// y axis
		y_g = g.append("g")
			.classed("y axis", true)
			.attr("transform", "translate(" + width + ",0)")

		xScale = d3.scaleBand()
			.rangeRound([0, width])
			.padding(.1);

		yScale = d3.scaleLinear()
			.range([height, 0])

		brushg = g.append("g")
			.classed("brush_snapping", true)
	}
	chart.update = update;
		
	function update() {
		brush_years.text(brushYearStart + " - " + brushYearEnd)
	
		var hist = d3.histogram()
			.value(function(d) { return d.year })
			.domain([brushYearStart, brushYearEnd])
			.thresholds(thresholds);
		var bins = hist(filteredData);

		console.log("Histogram bins", bins)

		// Scales
		yScale.domain([0, d3.max(bins, function(d) { return d.length})]);

		y_axis = d3.axisRight(yScale);

		y_g.call(y_axis);

		drawBins();
		
		brushg.selectAll("rect")
		 	.attr("height", height);

		function drawBins() {
			//remove old histogram
			rects_g.selectAll("rect").remove()

			// Add a rect for each date.
			rect_data = rects_g.selectAll("rect").data(bins)
		
			rect_data.enter()
				.append("rect")
				.classed("bar", true)
				.attr("transform", function(d) {return "translate(" + xScale(d.x0) + "," + yScale(d.length) + ")"; })
				.attr("height", function(d) { return height - yScale(d.length); })
				.attr("width", xScale.bandwidth())
				.style("fill", "#737f96")
		}
	}

	chart.highlightBin = function(point) {
		rect = rects_g.selectAll("rect")
			.select(function (d, i) { return d.x0 == point.year ? this : null})
			.style("fill", "#ffb20a")		
	}

	chart.removeHighlightBin = function(point) {
		rect = rects_g.selectAll("rect")
			.select(function (d, i) { return d.x0 == point.year ? this : null})
				.style("fill", "#737f96")
	}
	
	chart.highlight = function(node) {
		rectsOverlay_g.selectAll("rect.highlighted").remove();

		if (node.depth > filteredDepth) {
			highlightedData = filteredData.filter(function(d) { 
				return node.data.data.ids.includes(d.publication_number) 
			})
	
			var hist = d3.histogram()
				.value(function(d) { return d.year })
				.domain([brushYearStart, brushYearEnd])
				.thresholds(thresholds);
			var highlightedBins = hist(highlightedData);
			//console.log(highlightedBins);
	
			highlightedBinsData = rectsOverlay_g.selectAll("rect").data(highlightedBins)

			highlightedBinsData.enter()
				.append("rect")
				.classed("bar", true)
				.classed("highlighted", true)
				.attr("transform", function(d) {return "translate(" + xScale(d.x0) + "," + yScale(d.length) + ")"; })
				.attr("height", function(d) { return height - yScale(d.length); })
				.attr("width", xScale.bandwidth())
				.style("fill", node.color)				
		} 
	}

	chart.filter = function (node) {
		if (node.depth != filteredDepth) { //preventing circular dependencies
			filteredDepth = node.depth;

			if (node.depth == 0) {
				filteredData = data; //reset time selection when sunburst reset to root
				//reset filter
				brushend();
				//brushend(explicitlyInvoked = true);
			} else {
				var ids = node.data.data.ids;
				filteredData = data.filter(function(d) { return ids.includes(d.publication_number) ? this : null })
			}
			
			rectsOverlay_g.selectAll("rect.highlighted").remove();
			
			update();
		}		
	}

    chart.data = function(value) {
        if(!arguments.length) return data;
		data = value;
		filteredData = data;
		filteredDepth = 0;		

		prepareBrush();
			
		update();

		return chart;
		
		function prepareBrush() {
			brushYearStart = d3.min(data, function (d) { return d.year; });
			brushYearEnd = d3.max(data, function (d) { return d.year; });
			thresholds = [];
			for (i = brushYearStart; i <= brushYearEnd; i++) {
				thresholds.push(i);
			}
			xScale.domain(thresholds);
			// Axis variables for the bar chart		
			x_axis = d3.axisBottom(xScale)
				.tickValues(xScale.domain().filter(function (d, i) { return !(i % 2); })); //ticks every 2 years
			x_g.call(x_axis);
			// Draw the brush
			brush = d3.brushX()
				.extent([[xScale.range()[0], 0], [xScale.range()[1], height]])
				.on("brush", brushmove)
				.on("end", brushend);
			brushg.call(brush);
			drawHandles();
			brushg.select(".selection")
				.attr("fill", "#000")
				.style("display", "none");
		
			function drawHandles() {
				var handle_data = brushg.selectAll(".handle--custom")
					.data([{ type: "w" }, { type: "e" }], function (d) { return d.type; });
				handle_enter = handle_data.enter().append("path")
					.attr("class", "handle--custom")
					.attr("stroke", "#000")
					.attr("cursor", "ew-resize")
					.attr("d", brushResizePath);
				handle = handle_data.merge(handle_enter)
					.attr("transform", null)
					.attr("display", "none");
				function brushResizePath(d) {
					var e = +(d.type == "e"), x = e ? 1 : -1, y = height / 2;
					return "M" + (.5 * x) + "," + y + "A6,6 0 0 " + e + " " + (6.5 * x) + "," + (y + 6) + "V" + (2 * y - 6) + "A6,6 0 0 " + e + " " + (.5 * x) + "," + (2 * y) + "Z" + "M" + (2.5 * x) + "," + (y + 8) + "V" + (2 * y - 8) + "M" + (4.5 * x) + "," + (y + 8) + "V" + (2 * y - 8);
				}
			}
			
			function brushmove() {
		
				if (d3.event.sourceEvent.type === "brush") 
						return;
				
				var localBrushYearStart, localBrushYearEnd;
		
				if (d3.event.selection === null) {
						localBrushYearStart = brushYearStart;
						localBrushYearEnd = brushYearEnd;
						// Update start and end years in upper right-hand corner of the map
						brush_years.text(localBrushYearStart == localBrushYearEnd ? 
							localBrushYearStart : localBrushYearStart + " - " + localBrushYearEnd);
						handle.attr("display", "none");
						rects_g.selectAll("rect.bar").style("opacity", "1");
				} 
				else {
					var d0 = d3.event.selection.map(scaleBandInvert(xScale)); 
			
					d1 = d0;//d0.map(d3.timeYear.round);
		
					// If empty when rounded, use floor instead.
					if (d1[0] >= d1[1]) {
						d1[0] = d3.timeYear.floor(d0[0]);
						d1[1] = d3.timeYear.offset(d1[0]);
					}
		
					// Snap to rect edge
					d3.select(this).call(d3.event.target.move, d1.map(xScale));
		
					localBrushYearStart = d1[0];
					localBrushYearEnd = d1[1];
		
					if (localBrushYearStart == localBrushYearEnd){
						handle.attr("display", "none");
						rects_g.selectAll("rect.bar").style("opacity", "1");
					}
					else {                    
						handle.attr("display", null).attr("transform", function(d, i) { 
							x_positions = d1.map(xScale)[i];
							if (x_positions != null) {
								return "translate(" + [x_positions, - height / 4] + ")"; 
							}
							else
							{
								return "translate(" + [0, - height / 4] + ")"; 
							}
						});
						// Fade all years in the histogram not within the brush
						rects_g.selectAll("rect.bar").style("opacity", function(d) {
							return d.x0 >= localBrushYearStart && d.x0 < localBrushYearEnd ? "1" : ".4";
						});
					}
				}     
			}
		}
	}
	
	function brushend() {
		
		var localBrushYearStart, localBrushYearEnd;

		if (d3.event == null || d3.event.selection === null) {
			localBrushYearStart = brushYearStart;
			localBrushYearEnd = brushYearEnd;
			// Update start and end years in upper right-hand corner of the map
			brush_years.text(localBrushYearStart == localBrushYearEnd ? 
				localBrushYearStart : localBrushYearStart + " - " + localBrushYearEnd);

			handle.attr("display", "none");
			brushg.select(".selection")
				.attr("fill", "#000")
				.style("display", "none");
			rects_g.selectAll("rect.bar").style("opacity", "1");
			
			//reset filter
			dispatch.call("filter", this, data)			
		} 
		else {
			var d0 = d3.event.selection.map(scaleBandInvert(xScale)); 
	
			d1 = d0;

			// If empty when rounded, use floor instead.
			if (d1[0] >= d1[1]) {
					d1[0] = d3.timeYear.floor(d0[0]);
					d1[1] = d3.timeYear.offset(d1[0]);
			}

			localBrushYearStart = d1[0],
			localBrushYearEnd =  d1[1];

			// Fade all years in the histogram not within the brush
			rects_g.selectAll("rect.bar").style("opacity", function(d) {
					return d.x0 >= localBrushYearStart && d.x0 < localBrushYearEnd ? "1" : ".4";
			});
			// Update start and end years in upper right-hand corner of the map
			brush_years.text(localBrushYearStart == (localBrushYearEnd-1) ? 
				localBrushYearStart : localBrushYearStart + " - " + (localBrushYearEnd-1));

			//restrict data points to selected years
			var filtered = filteredData.filter(function(d) {
				return (d.year >= localBrushYearStart && d.year <= localBrushYearEnd)
			})

			//emit filtered data
			dispatch.call("filter", this, filtered)
		}   
	}

	function scaleBandInvert(scale) {
		var domain = scale.domain();
		var paddingOuter = scale(domain[0]);
		var eachBand = scale.step();
		return function (value) {
			var index = Math.floor(((value - paddingOuter) / eachBand));
			return domain[Math.max(0, Math.min(index, domain.length-1))];
		}
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
    
    return d3.rebind(chart, dispatch, "on");
}


