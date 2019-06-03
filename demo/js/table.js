if(!d3.chart) d3.chart = {};

d3.chart.table = function() {
	var data;
	var width;
	var height;
	var table;
	var dispatch = d3.dispatch();
	
	function chart(container) {
		table = container.append("xhtml:table")
			.attr("max-width", width)
			.attr("width", "100%")
			.attr("max-height", height)
			.on("dblclick", showFullInformation);			
	}
	chart.update = update;

	function update() {
		var table_data = table.datum(data);

		table_data.selectAll("tr").remove();

		var header_row = table_data.append("tr")		

		header_row.append("th")
			.attr("width", 90)
			.text(function (d) { return d.publication_number })
			
		header_row.append("th")
			.attr("align", "left")
			.text(function (d) { return d.title_text })
			
		var desc_row = table_data.append("tr")		
		desc_row.append("td")
			.text(function (d) { return d3.timeFormat("%Y.%m.%d")(d.priority_date) });
		desc_row.append("td")
			.text(function (d) { return d.raw_assignees.join(", ") });

		var citation_row = table_data.append("tr")		
			citation_row.append("td")
				.text(function (d) { return "Cites " + d.references.length });
			citation_row.append("td")
				.text(function (d) { return "Cited by " + d.referenced_by.length + " in this dataset"});

		var class_row = table_data.append("tr")					
		class_row.append("td")
			.attr("colspan", 2)	
			.style("padding", '2px 0px')
			.text(function (d) { return d.ipc_classes.join(", ") })

		var term_row = table_data.append("tr")		
		term_row.append("td")
			.attr("colspan", 2)
			.style("padding", '4px 0px')
			.text(function (d) { return d.terms.slice(0, 10).join(", ") })

		var abstract_row = table_data.append("tr")
			.style("max-height", "140px")
			.style("padding", '4px 0px')

		abstract_row.append("td")
			.attr("class", "abstract")
			.attr("colspan", 2)
			.style("max-height", "140px")
			.append("div")
				.style("max-height", "140px")
				.styles({overflow: "auto",
						width: "100%"})
				.text(function (d) { return d.abstract_text })			
	}

	function showFullInformation(d) {

		var abstract_text = "<b>Abstract:</b><br>" + d.abstract_text + "<hr><br>";

		var claims_text = "<b>Claims:</b><br>" + d.claims_text.replace("\n", '<br>');
		if (d.claims_text == "None") {
			claims_text = "No claims available for this patent."
		}		

		var win = window.open("", "", 
			"toolbar=no,location=no,directories=no,status=no,menubar=no,scrollbars=yes,resizable=yes," + 
			"width=500,height=600,top=0,left=" + (screen.width - 860));
			win.document.title = d.title_text;
			win.document.body.innerHTML = abstract_text + claims_text;

			// fScript = document.createElement('script');
			 
			// function injectThis() {
			// 	self.moveTo(500, 0); 
			// 	self.focus();
			// }
			//fScript.innerHTML = "window.onblur = " + injectThis.toString() + ";";
			//win.document.body.appendChild(fScript)
		win.focus();
	}
 
	chart.showDetails = function(hovered) {
		if (hovered != null){
			data = hovered;
			update();
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