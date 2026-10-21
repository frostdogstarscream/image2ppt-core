import fs from 'node:fs/promises';

// Authoring one page of a presentation from a loaded SlideIR.
//
// This is the per-slide half of the compiler, shared by the single-page
// `compile` path and the multi-page `compose` path so both author a page
// identically. The single-page caller passes no label, so its output is
// unchanged by this extraction.
export async function buildSlide(presentation, {ir, info}, pageLabel) {
  const [w,h]=ir.canvas.source_px,[tw,th]=ir.canvas.target_px;
  const slide=presentation.slides.add();
  slide.background.fill='#'+ir.background.color;
  const position=box=>({left:box[0]/w*tw,top:box[1]/h*th,width:box[2]/w*tw,height:box[3]/h*th});
  async function addImage(id,asset,bbox) {
    const bytes=await fs.readFile(asset._file);
    const image=slide.images.add({name:id,blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),
      contentType:asset._file.toLowerCase().endsWith('.png')?'image/png':'image/jpeg',
      alt:id,fit:'contain',position:position(bbox)});
    image.name=id;
  }
  if (ir.background.asset_id) await addImage('background-image',info.assets.get(ir.background.asset_id),[0,0,w,h]);
  for (const node of info.ordered) {
    if (node.type==='image') { await addImage(node.id,info.assets.get(node.asset_id),node.bbox); continue; }
    if (node.type==='table') {
      // The name must be set on the handle: passing `name` in the add options is
      // silently ignored, and the OOXML adapter looks objects up by it.
      const table=slide.tables.add({rows:node.rows.length,columns:node.rows[0].length,...position(node.bbox),
        values:node.rows,
        ...(node.column_tracks?{columnTracks:node.column_tracks.map(t=>({mode:t.mode,value:t.value}))}:{})});
      if (node.header_row!==undefined || node.banded_rows!==undefined) {
        table.styleOptions={...(node.header_row!==undefined?{headerRow:node.header_row}:{}),
          ...(node.banded_rows!==undefined?{bandedRows:node.banded_rows}:{})};
      }
      table.name=node.id;
      // Cells default to Calibri otherwise, which the layout validator rejects
      // as an unapproved family; the declared font is the author's choice.
      // The cell style key is `typeface`; passing latin/ea here is silently
      // ignored, and `fontSize` is in pixels.
      table.cells.block({row:0,column:0,rowCount:node.rows.length,columnCount:node.rows[0].length})
        .assign({textStyle:{typeface:node.font.ea,fontSize:node.font.size_pt*96/72}});
      if(node.row_heights) node.row_heights.forEach((height,index)=>{table.rows[index].height=height;});
      for(const m of node.merges||[]) table.merge({startRow:m.row,endRow:m.row+m.row_span-1,startColumn:m.column,endColumn:m.column+m.column_span-1});
      for(const s of node.cell_styles||[]) table.cells.block({row:s.row,column:s.column,rowCount:1,columnCount:1})
        .assign({...(s.fill!==undefined?{fill:'#'+s.fill}:{}),textStyle:{typeface:node.font.ea,fontSize:node.font.size_pt*96/72,
          ...(s.color!==undefined?{color:'#'+s.color}:{}),...(s.bold!==undefined?{bold:s.bold}:{})}});
      continue;
    }
    if (node.type==='chart') {
      // Name goes on the handle, not in the options — see the table branch.
      const typeface={typeface:node.font.ea};
      const chart=slide.charts.add(node.chart_type,{position:position(node.bbox),
        ...(node.title!==undefined?{title:node.title,titleTextStyle:typeface}:{}),
        categories:node.categories,
        series:node.series.map(s=>({name:s.name,values:s.values})),
        xAxis:{textStyle:typeface},yAxis:{textStyle:typeface},
        ...(node.has_legend!==undefined?{hasLegend:node.has_legend}:{}),
        ...(node.legend_position?{legend:{position:node.legend_position,textStyle:typeface}}
          :node.has_legend?{legend:{textStyle:typeface}}:{})});
      chart.name=node.id;
      continue;
    }
    const shape=slide.shapes.add({geometry:node.type==='text'?'textbox':node.geometry,name:node.id,position:position(node.bbox),
      fill:node.type==='text' || node.fill==='none' ? 'none' : node.fill_gradient ? {
        type:'gradient',gradientKind:'linear',angleDeg:node.fill_gradient.angle_deg,
        stops:node.fill_gradient.stops.map(s=>({offset:s.offset,color:'#'+s.color}))
      } : '#'+node.fill,
      ...(node.corner_radius_px!==undefined?{borderRadius:node.corner_radius_px/w*tw}:{}),
      line:node.line?{style:'solid',fill:'#'+node.line.color,width:node.line.width_pt*96/72}:{fill:'none',width:0}});
    if (node.type==='text') {
      shape.text=node.runs ? [node.runs.map(r=>({run:r.text,style:{bold:r.bold??node.bold??false,color:'#'+(r.color||node.color)}}))] : node.text;
      shape.text.style={typeface:node.font.ea,fontSize:node.font.size_pt*96/72,bold:node.bold??false,
        color:'#'+node.color,alignment:node.align||'left',verticalAlignment:node.vertical_anchor||'top',
        wrap:node.wrap?'square':'none',autoFit:'none',insets:0,
        ...(node.line_spacing_pt?{lineSpacing:node.line_spacing_pt*96/72}:{})};
    }
  }
  slide.speakerNotes.text='Source SHA256: '+ir.source_sha256+(pageLabel?' | '+pageLabel:'');
  return slide;
}
