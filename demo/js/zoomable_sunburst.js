if(!d3.chart) d3.chart = {};

d3.chart.sunburst = function() {

    var g;
    var data;
    var width = 400;
    var height = 500;
    const CHAR_SPACE = 7;
    const breadcrumbHeight = 25;
    const explanationHeight = 90;
    var sunburstArea;
    var radius;
    var xScale, yScale;
    var formatNumber;
    var currentTopNode;
    var hierarchy;
    var explanations;

    var dispatch = d3.dispatch("hover", "sunburstLevelChanged");

    function chart(container) {
        g = container;
        
        radius = Math.min(width, height - breadcrumbHeight - explanationHeight - 30) / 2;

        // setting up breadcrumbs view
        // Add the svg area.
        var trail = g.append("g")
            .attr("id", "trail")
            .attr("height", breadcrumbHeight + 20 + explanationHeight)
            .attr("width", width)
            .attr("transform", "translate(" + [0, 10] + ")")

        trail.append("text")
            .attr("id", "explanation")
            .attr("transform", "translate(" + [0, 20 + breadcrumbHeight] + ")")

        sunburstArea = g.append("g")
            .attr("id", "sunburst")
            .attr("transform", "translate("+ [width / 2, breadcrumbHeight + explanationHeight + 27 + radius] + ")")
       
        // Bounding circle underneath the sunburst, to make it easier to detect
        // when the mouse leaves the parent g.
        sunburstArea.append("circle")
            .attr("r", radius)
            .attr("fill", "#f6f6f6")       
            //.style("opacity", 0.3);

        var maxRadius = radius - 5;

        xScale = d3.scaleLinear()
            .range([0, 2 * Math.PI])
            .clamp(true);

        yScale = d3.scalePow()
            .exponent(0.7)
            .range([maxRadius*.1, maxRadius]);

        formatNumber = d3.format(',d');
    }
    chart.update = update;
      
    function update() {

        const partition = d3.partition();

        const arc = d3.arc()
            .startAngle(d => xScale(d.x0))
            .endAngle(d => xScale(d.x1))
            .innerRadius(d => Math.max(0, yScale(d.y0)))
            .outerRadius(d => Math.max(0, yScale(d.y1)));

        function getColor(d) {

            if (d.depth <= currentTopNode.depth + 1) {
                mid_angle = xScale((d.x0 + d.x1) / 2)
                mid = mid_angle / (Math.PI * 2)

                const epsilon = 0.0001

                if (mid < 0.5 + epsilon && mid > 0.5 - epsilon) {
                    d.color = d3.rgb(115, 127, 150)
                } else {
                    d.color = d3.interpolateRainbow(mid)                                                       
                }

                //console.log(d.data.data.name, d.x0, d.x1, mid_angle, mid, d.color)                       
            }        
            
            if (d.children) {
                var startColor = d3.cubehelix(d.color).darker();
                var endColor   = d3.cubehelix(d.color).brighter();

                // Create the scale
                colors = d3.scaleLinear()
                    .interpolate(d3.interpolateCubehelixLong)
                    .range([startColor.toString(), endColor.toString()])
                    .domain([0, d.children.length + 1]);

                d.children.map(function(child, i) {
                        return {value: child.value, idx: i};
                    }).sort(function(a, b) {
                        return b.value - a.value
                    }).forEach(function(child, i) {
                        d.children[child.idx].color = colors(i);
                    });                    
            }        
         
            return d.color;
        }

        function middleArcLine(d) {
            const halfPi = Math.PI/2;
            const angles = [xScale(d.x0) - halfPi, xScale(d.x1) - halfPi];
            const r = Math.max(0, (yScale(d.y0) + yScale(d.y1)) / 2);

            const midAngle = (angles[1] + angles[0]) / 2;
            const invertDirection = midAngle > 0 && midAngle < Math.PI; // On lower quadrants write text ccw
            if (invertDirection) { angles.reverse(); }

            const path = d3.path();
            path.arc(0, 0, r, angles[0], angles[1], invertDirection);
            return path.toString();
        };

        function textFits(d) {
            if (d != null) {
                const deltaAngle = xScale(d.x1) - xScale(d.x0);
                const r = Math.max(0, (yScale(d.y0) + yScale(d.y1)) / 2);
                const perimeter = r * deltaAngle;

                return d.data.data.name.length * CHAR_SPACE < perimeter;
            }
            else return false;            
        };

        drawSlices();
        
        // Add the mouseleave handler to the bounding circle.
        sunburstArea.on("mouseleave", mouseleave);        
        sunburstArea.on('click', focusOn);

        function drawSlices() {
            
            currentTopNode = hierarchy;

            //removing entries from another hierarchy
            sunburstArea.selectAll('g.slice').remove();

            var slice = sunburstArea.selectAll('g.slice')
                .data(partition(hierarchy).descendants(), function(d) { return d.data.id });

            var newSlice = slice.enter()
                .append('g')
                .attr('class', 'slice')
                .attr("display", function(d) { return d.depth ? null : "none"; })
                .on('click', d => {
                    d3.event.stopPropagation();
                    focusOn(d);               
                })
                
            newSlice.append('title')
                .text(d => d.data.data.name + '\n' + formatNumber(d.data.data.ids.length));

            newSlice.append('path')
                .attr('class', 'main-arc')
                .attr("fill", getColor)
                .attr('d', arc);

            newSlice.append('path')
                .attr('class', 'hidden-arc')
                .attr('id', (_, i) => `hiddenArc${i}`)
                .attr('d', middleArcLine);

            var text = newSlice.append('text')
                .attr('display', d => textFits(d) ? null : 'none');

            // Add white contour
            text.append('textPath')
                .attr('startOffset','50%')
                .attr('xlink:href', (_, i) => `#hiddenArc${i}` )
                .text(d => d.data.data.name)
                .style('fill', 'none')
                .style('stroke', '#fff')
                .style('stroke-width', 5)
                .style('stroke-linejoin', 'round');

            text.append('textPath')
                .attr('startOffset','50%')
                .attr('xlink:href', (_, i) => `#hiddenArc${i}` )
                .text(d => d.data.data.name);
            
            var slices = sunburstArea.selectAll('g.slice')

            slices.on("mouseenter", mouseenter)

            focusOn();
        }
       
        function focusOn(d = hierarchy) { // Reset to top-level if no data point specified

            const transition = sunburstArea.transition("ZoomingInOrOut")
                .duration(500)
                .tween('scale', () => {
                    const xd = d3.interpolate(xScale.domain(), [d.x0, d.x1]);
                    const yd = d3.interpolate(yScale.domain(), [d.y0, 1]);                    
                    return  t => { 
                        xScale.domain(xd(t));                         
                        yScale.domain(yd(t));   
                        //console.log("x domain", x.domain())                      
                    };
                })

            transition.selectAll('path.main-arc')
                .attrTween('d', d => () => arc(d))
                .attrTween("fill", d =>() => getColor(d))

            transition.selectAll('path.hidden-arc')
                .attrTween('d', d => () => middleArcLine(d));

            transition.selectAll('text')
                .attrTween('display', d => () => textFits(d) ? null : 'none');

            moveStackToFront(d);

            transition.on("end", onSunburstLevelChanged)
            mouseleave();
           
            function moveStackToFront(elD) {
                sunburstArea.selectAll('.slice').filter(d => d === elD)
                    .each(function(d) {
                        this.parentNode.appendChild(this);
                        if (d.parent) { moveStackToFront(d.parent); }
                    })
                currentTopNode = elD;
            }
        }

        // Update the breadcrumb trail to show the current sequence and percentage.
        function updateBreadcrumbs(nodeArray, percentageString) {

            breadcrumbArray = nodeArray.map(function(d) {
                var text = d.data.data.name;
                text = text.length > 10 ? text.slice(0, 9) + "..." : text;
                var color = getColor(d);
                return {"text": text, "fill": color, "id": d.data.id};
            })

            var breadcrumb = d3.breadcrumb()
                .container('#trail')   
                .padding(2)
                .height(24)
                .fontSize(12)
                .marginLeft(0)
                .marginTop(10)
                //.wrapWidth(width - 50 - 40 - 30)
                .leftDirection(false)
            
            d3.select(".breadcrumb-trail").selectAll(".breadcrumbs").remove();
            breadcrumb.show(breadcrumbArray); 
        
            // Add the label at the end, for the percentage.
            var lastBreadcrumb = d3.select(".breadcrumb-trail").selectAll(".breadcrumbs").filter(":last-child")
            lastBreadcrumb.append("text")
                .attr("id", "endlabel")
                .attr("transform", "translate(" + (lastBreadcrumb.node().getBBox().width + 5) + "," + 17 + ")" )
                .text(percentageString);

            var explanationText = getExplanationText(nodeArray)

            d3.select("#explanation")
                .text(explanationText)
                .attr("x", 0)
                .attr("y", 0)
                .call(wrap, width - 10); 
        
            // Make the breadcrumb trail visible, if it's hidden.
            d3.select("#trail")
                .style("visibility", "");       

            function getExplanationText(arr) {
                var text = ""
                for (i = 0; i < arr.length - 1; i++) {
                    text = text + lookupExplanation(arr[i].data.data.name) + " > "
                }
                text = text + lookupExplanation(arr[arr.length - 1].data.data.name)
                return text;

                function lookupExplanation(key) {
                    return key in explanations ? explanations[key]["explanation"] : key;
                }
            }

            function wrap(text, width) {
                text.each(function () {
                    var text = d3.select(this),
                        words = text.text().split(" ").reverse(),
                        word,
                        line = [],
                        lineNumber = 0,
                        lineHeight = 1.2, // ems
                        x = text.attr("x"),
                        y = text.attr("y"),
                        dy = 0,
                        tspan = text.text(null)
                                    .append("tspan")
                                    .attr("x", x)
                                    .attr("y", y)
                                    .attr("dy", dy + "em");
                    while (word = words.pop()) {
                        line.push(word);
                        tspan.text(line.join(" "));
                        if (tspan.node().getComputedTextLength() > width) {
                            line.pop();
                            tspan.text(line.join(" "));
                            line = [word];
                            tspan = text.append("tspan")
                                        .attr("x", x)
                                        .attr("y", y)
                                        .attr("dy", ++lineNumber * lineHeight + dy + "em")
                                        .text(word);
                        }
                    }
                });
            }
        }

        // Fade all but the current sequence, and show it in the breadcrumb trail.
        function mouseenter(d) {

            if (d != null) {
                var sequenceArray = prepareAndUpdateBreadcrumbs(d, currentTopNode)

                // Fade all the segments.
                sunburstArea.selectAll("path")
                    .style("opacity", 0.3);
            
                // Then highlight only those that are an ancestor of the current segment.
                sunburstArea.selectAll("path")
                    .filter(function(node) {
                        return (sequenceArray.indexOf(node) >= 0);
                    })
                    .style("opacity", 1);
    
                dispatch.call("hover", this, d)
            }
        }
        
        function prepareAndUpdateBreadcrumbs(node, whole) {
            var sequenceArray = node.ancestors().reverse();
            sequenceArray.shift()

            var percentageString = computePercentageString(node, whole);

            updateBreadcrumbs(sequenceArray, percentageString);

            return sequenceArray;

            function computePercentageString(part, whole) {
                var percentage = (100 * part.data.data.ids.length / whole.data.data.ids.length).toPrecision(3);
                    var percentageString = percentage + "%";
                    if (percentage < 0.1) {
                        percentageString = "< 0.1%";
                    }
                return percentageString
            }
        }

        // Restore everything to full opacity when moving off the visualization.
        function mouseleave(d) {
        
            if (currentTopNode.depth > 0) {
                var realTopNode;
                //currentTopNode does not have a parent cause it was ripped from hierarchy.
                //looking for it in the hierarchy
                hierarchy.each(function(node) { 
                    if (node.data.id == currentTopNode.data.id) {                  
                        realTopNode = node;                        
                    }                
                })

                prepareAndUpdateBreadcrumbs(realTopNode, hierarchy)
            }
            else {
                // Hide the breadcrumb trail
                d3.select("#trail")
                .style("visibility", "hidden");
            }            
        
            // Deactivate all segments during transition.
            sunburstArea.selectAll("path").on("mouseenter", null);
        
            // Transition each segment to full opacity and then reactivate it.
            sunburstArea.selectAll("path")
                .transition()
                .duration(750)
                .style("opacity", 1)
                .on("end", function() {
                    d3.select(this).on("mouseenter", mouseenter);
                });
            
            dispatch.call("hover", this, currentTopNode) 
        }
    }

    chart.filter = function(filtered) {
        ids = filtered.map(d => d.publication_number);

        var filteredData = data.map(function (d) {
            filteredIds = d.ids.filter(d => ids.includes(d));

            if (filteredIds.length > 0) {
                newD = JSON.parse(JSON.stringify(d))
                newD.ids = filteredIds;
                return newD;
            } else {
                return null;
            }
        })
        filteredData = filteredData.filter(d => {return d != null})

        tempHierarchy = createHierarchy(filteredData);
        if (tempHierarchy.value != hierarchy.value) {
            hierarchy = tempHierarchy;
            update();
        }
        
        return hierarchy;
    }

    chart.data = function(value) {
        if(!arguments.length) return data;

        data = value;      
        
        hierarchy = createHierarchy(data);

        update();

        return chart;
    }

    chart.explanations = function(value) {
        if (!arguments.length) return explanations;
        explanations = value;
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
    
    function createHierarchy(data) {
        var root = d3.stratify()
            .id(function(d) { return d.id; })						
            .parentId(function(d) { return d.parent; })
        (data)

        root.each(function(node) { node.value = normalizeSector(node) });

        var hierarchy = d3.hierarchy(root);						
        hierarchy.sort(function(a, b) { 
            return b.value - a.value; 
        });

        console.log("hierarchy", hierarchy)
        return hierarchy;

        function normalizeSector(d) { 
            if (d.parent){
                sum_ids_siblings = 0;
                
                for (i = 0; i < d.parent.children.length; i++) { 
                    child = d.parent.children[i];
                    sum_ids_siblings += child.data.ids.length;
                } 
                return d.data.ids.length / sum_ids_siblings * d.parent.value
            } else {
                return d.data.ids.length
        }}
    }

    function onSunburstLevelChanged() {
        dispatch.call("sunburstLevelChanged", this, [currentTopNode, xScale.domain()])
    }

    return d3.rebind(chart, dispatch, "on");
}