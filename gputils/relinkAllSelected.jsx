#target Illustrator

// script.name = relinkAllSelected.jsx;

// script.description = relinks all selected placed images at once;

// script.required = select at least one linked image before running;

// script.parent = CarlosCanto // 7/12/11;

// script.elegant = false;

var idoc = app.activeDocument;

sel = idoc.selection;

// alert(sel.length);

///*
if (sel.length>0){
	var file = File(sel[0].file).openDlg(); //File.openDialog("open file " + iplaced.file);
	var dirname = file.fsName.match(/(.*)[\/\\]/)[1]||'';
	for (i=0 ; i<sel.length ; i++ ){
		//alert(sel[i].typename)
		if (sel[i].typename == "PlacedItem"){
			var iplaced = sel[i];
			
			//var file_new = 
			var filename1 = iplaced.file.fsName.split('\\').pop();
			//filename1 = filename1.split('_').pop();
			//alert(filename1)
			var newfilepath = dirname + '\\' + filename1
			
			//alert()
			iplaced.file = File(newfilepath);
		}
	}
}
//*/
/*
if (sel.length>0){
	//var file = File(sel[0].file).openDlg(); //File.openDialog("open file " + iplaced.file);
	//var dirname = file.fsName.match(/(.*)[\/\\]/)[1]||'';
	for (i=0 ; i<sel.length ; i++ ){
		//alert(sel[i].typename)
		if (sel[i].typename == "PlacedItem"){
			var iplaced = sel[i];
			
			//var file_new = 
			var dirname = iplaced.file.fsName.match(/(.*)[\/\\]/)[1]||'';
			var filename1 = iplaced.file.fsName.split('\\').pop().split('_').shift();
			//filename1 = filename1.split('_').pop();
			//alert(filename1)
			var newfilepath = dirname + '\\' + filename1 + '_gray_depth.png'
			
			//alert()
			iplaced.file = File(newfilepath);
		}
	}
}
else {
	alert("select at least one placed item before running");
} 
*/