import{_ as a,c as e,o as t,V as o}from"./chunks/framework.b1Lc_bjX.js";const f=JSON.parse('{"title":"Integrations","description":"","frontmatter":{},"headers":[],"relativePath":"integrations.md","filePath":"integrations.md","lastUpdated":null}'),r={name:"integrations.md"},i=o('<h1 id="4351147819250033906" tabindex="-1">Integrations <a class="header-anchor" href="#4351147819250033906" aria-label="Permalink to &quot;Integrations {#4351147819250033906}&quot;">​</a></h1><h2 id="17497016727470127283" tabindex="-1">Spatial sciences <a class="header-anchor" href="#17497016727470127283" aria-label="Permalink to &quot;Spatial sciences {#17497016727470127283}&quot;">​</a></h2><h3 id="5121625197889341269" tabindex="-1">Rasters.jl <a class="header-anchor" href="#5121625197889341269" aria-label="Permalink to &quot;Rasters.jl {#5121625197889341269}&quot;">​</a></h3><p><a href="https://rafaqz.github.io/Rasters.jl/stable">Raster.jl</a> extends DD for geospatial data manipulation, providing file load/save for a wide range of raster data sources and common GIS tools like polygon rasterization and masking. <code>Raster</code> types are aware of <code>crs</code> and their <code>missingval</code> (which is often not <code>missing</code> for performance and storage reasons).</p><p>Rasters.jl is also the reason DimensionalData.jl exists at all! But it always made sense to separate out spatial indexing from GIS tools and dependencies.</p><p>A <code>Raster</code> is a <code>AbstractDimArray</code>, a <code>RasterStack</code> is a <code>AbstractDimStack</code>, and <code>Projected</code> and <code>Mapped</code> are <code>AbstractSample</code> lookups.</p><h3 id="383357156870636445" tabindex="-1">YAXArrays.jl <a class="header-anchor" href="#383357156870636445" aria-label="Permalink to &quot;YAXArrays.jl {#383357156870636445}&quot;">​</a></h3><p><a href="https://juliadatacubes.github.io/YAXArrays.jl/dev/">YAXArrays.jl</a> is another spatial data package aimmed more at (very) large datasets. It&#39;s functionality is slowly converging with Rasters.jl (both wrapping DiskArray.jl/DimensionalData.jl) and we work closely with the developers.</p><p><code>YAXArray</code> is a <code>AbstractDimArray</code> and inherits its behaviours.</p><h3 id="5395315509484401575" tabindex="-1">ClimateBase.jl <a class="header-anchor" href="#5395315509484401575" aria-label="Permalink to &quot;ClimateBase.jl {#5395315509484401575}&quot;">​</a></h3><p><a href="https://juliaclimate.github.io/ClimateBase.jl/dev/">ClimateBase.jl</a> Extends DD with methods for analysis of climate data.</p><h2 id="7767869655895230401" tabindex="-1">Statistics <a class="header-anchor" href="#7767869655895230401" aria-label="Permalink to &quot;Statistics {#7767869655895230401}&quot;">​</a></h2><h2 id="12225069691595720931" tabindex="-1">ArviZ.jl <a class="header-anchor" href="#12225069691595720931" aria-label="Permalink to &quot;ArviZ.jl {#12225069691595720931}&quot;">​</a></h2><p><a href="https://arviz-devs.github.io/ArviZ.jl/dev/">ArviZ.jl</a> Is a julia package for exploratory analysis of Bayesian models.</p><p>An <code>ArviZ.Dataset</code> is an <code>AbstractDimStack</code>!</p><h2 id="1047670445670295111" tabindex="-1">Optimization <a class="header-anchor" href="#1047670445670295111" aria-label="Permalink to &quot;Optimization {#1047670445670295111}&quot;">​</a></h2><h3 id="6415149364467815075" tabindex="-1">JuMP.jl <a class="header-anchor" href="#6415149364467815075" aria-label="Permalink to &quot;JuMP.jl {#6415149364467815075}&quot;">​</a></h3><p><a href="https://jump.dev/">JuMP.jl</a> is a powerful omptimisation DSL. It defines its own named array types but now accepts any <code>AbstractDimArray</code> too, through a package extension.</p><h2 id="782416527003373519" tabindex="-1">Simulations <a class="header-anchor" href="#782416527003373519" aria-label="Permalink to &quot;Simulations {#782416527003373519}&quot;">​</a></h2><h3 id="11492899724619659527" tabindex="-1">CryoGrid.jl <a class="header-anchor" href="#11492899724619659527" aria-label="Permalink to &quot;CryoGrid.jl {#11492899724619659527}&quot;">​</a></h3><p><a href="https://juliahub.com/ui/Packages/General/CryoGrid">CryoGrid.jl</a> A Juia implementation of the CryoGrid permafrost model.</p><p><code>CryoGridOutput</code> uses <code>DimArray</code> for views into output data.</p><h3 id="8258062442671439922" tabindex="-1">DynamicGrids.jl <a class="header-anchor" href="#8258062442671439922" aria-label="Permalink to &quot;DynamicGrids.jl {#8258062442671439922}&quot;">​</a></h3><p><a href="https://github.com/cesaraustralia/DynamicGrids.jl">DynamicGrids.jl</a> is a spatial simulation engine, for cellular automata and spatial process models.</p><p>All DynamicGrids.jl <code>Outputs</code> are <code>&lt;: AbstractDimArray</code>, and <code>AbstractDimArray</code> are used for auxiliary data to allow temporal synchronisation during simulations. Notably, this all works on GPUs!</p><h2 id="18035618408509520877" tabindex="-1">Analysis <a class="header-anchor" href="#18035618408509520877" aria-label="Permalink to &quot;Analysis {#18035618408509520877}&quot;">​</a></h2><h3 id="8267433935677461594" tabindex="-1">AstroImages.jl <a class="header-anchor" href="#8267433935677461594" aria-label="Permalink to &quot;AstroImages.jl {#8267433935677461594}&quot;">​</a></h3><p><a href="http://juliaastro.org/dev/modules/AstroImages/">AstroImages.jl</a> Provides tools to load and visualise astromical images. <code>AstroImage</code> is <code>&lt;: AbstractDimArray</code>.</p><h3 id="17860640494152905021" tabindex="-1">TimeseriesTools.jl <a class="header-anchor" href="#17860640494152905021" aria-label="Permalink to &quot;TimeseriesTools.jl {#17860640494152905021}&quot;">​</a></h3><p>[TimeseriesTools.jl](<a href="https://juliahub.com/ui/Packages/General/TimeseriesTools" target="_blank" rel="noreferrer">https://juliahub.com/ui/Packages/General/TimeseriesTools</a> Uses <code>DimArray</code> for time-series data.</p>',30),s=[i];function l(d,n,c,h,m,p){return t(),e("div",null,s)}const b=a(r,[["render",l]]);export{f as __pageData,b as default};