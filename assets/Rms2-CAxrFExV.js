import{r as a,b as ae,j as t,a as rn}from"./index-0QIkLdMn.js";import{u as Ae,r as tt}from"./data-DxumdiML.js";import"./bf16-dzl8d6tk.js";import{H as on,a as ut,u as Ut}from"./charts-Dzf0QrkI.js";import{L as it,b as je,c as Se,S as sn}from"./ui-bjIF31__.js";import{R as an,g as ln,F as mt,h as nt,i as ge,W as cn,B as Me,j as Nt,V as w,b as dn,U as ft,k as pt,l as Wt,m as un,n as ye,L as mn,M as $,o as It,p as le,q as fn,_ as rt,Q as at,H as Oe,D as Ue,r as $t,s as Ht,d as Gt,S as qt,C as Vt,A as Xt,T as Kt,c as ie,t as pn,v as hn}from"./three-CFpLbQaf.js";import{u as xn,M as gn,C as yn,a as vn,E as wn,L as Ne,P as We,T as ht}from"./index-BkOLge58.js";import{c as Re,b as I,r as lt,d as bn}from"./tensor-Bb3R1QvN.js";import{C as Qt}from"./cells-rPQmNSsS.js";const Yt=parseInt(an.replace(/\D+/g,"")),Zt=Yt>=125?"uv1":"uv2",xt=new Me,ke=new w;class ct extends ln{constructor(){super(),this.isLineSegmentsGeometry=!0,this.type="LineSegmentsGeometry";const e=[-1,2,0,1,2,0,-1,1,0,1,1,0,-1,0,0,1,0,0,-1,-1,0,1,-1,0],r=[-1,2,1,2,-1,1,1,1,-1,-1,1,-1,-1,-2,1,-2],s=[0,2,1,2,3,1,2,4,3,4,5,3,4,6,5,6,7,5];this.setIndex(s),this.setAttribute("position",new mt(e,3)),this.setAttribute("uv",new mt(r,2))}applyMatrix4(e){const r=this.attributes.instanceStart,s=this.attributes.instanceEnd;return r!==void 0&&(r.applyMatrix4(e),s.applyMatrix4(e),r.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}setPositions(e){let r;e instanceof Float32Array?r=e:Array.isArray(e)&&(r=new Float32Array(e));const s=new nt(r,6,1);return this.setAttribute("instanceStart",new ge(s,3,0)),this.setAttribute("instanceEnd",new ge(s,3,3)),this.computeBoundingBox(),this.computeBoundingSphere(),this}setColors(e,r=3){let s;e instanceof Float32Array?s=e:Array.isArray(e)&&(s=new Float32Array(e));const o=new nt(s,r*2,1);return this.setAttribute("instanceColorStart",new ge(o,r,0)),this.setAttribute("instanceColorEnd",new ge(o,r,r)),this}fromWireframeGeometry(e){return this.setPositions(e.attributes.position.array),this}fromEdgesGeometry(e){return this.setPositions(e.attributes.position.array),this}fromMesh(e){return this.fromWireframeGeometry(new cn(e.geometry)),this}fromLineSegments(e){const r=e.geometry;return this.setPositions(r.attributes.position.array),this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new Me);const e=this.attributes.instanceStart,r=this.attributes.instanceEnd;e!==void 0&&r!==void 0&&(this.boundingBox.setFromBufferAttribute(e),xt.setFromBufferAttribute(r),this.boundingBox.union(xt))}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Nt),this.boundingBox===null&&this.computeBoundingBox();const e=this.attributes.instanceStart,r=this.attributes.instanceEnd;if(e!==void 0&&r!==void 0){const s=this.boundingSphere.center;this.boundingBox.getCenter(s);let o=0;for(let i=0,c=e.count;i<c;i++)ke.fromBufferAttribute(e,i),o=Math.max(o,s.distanceToSquared(ke)),ke.fromBufferAttribute(r,i),o=Math.max(o,s.distanceToSquared(ke));this.boundingSphere.radius=Math.sqrt(o),isNaN(this.boundingSphere.radius)&&console.error("THREE.LineSegmentsGeometry.computeBoundingSphere(): Computed radius is NaN. The instanced position data is likely to have NaN values.",this)}}toJSON(){}applyMatrix(e){return console.warn("THREE.LineSegmentsGeometry: applyMatrix() has been renamed to applyMatrix4()."),this.applyMatrix4(e)}}class Jt extends ct{constructor(){super(),this.isLineGeometry=!0,this.type="LineGeometry"}setPositions(e){const r=e.length-3,s=new Float32Array(2*r);for(let o=0;o<r;o+=3)s[2*o]=e[o],s[2*o+1]=e[o+1],s[2*o+2]=e[o+2],s[2*o+3]=e[o+3],s[2*o+4]=e[o+4],s[2*o+5]=e[o+5];return super.setPositions(s),this}setColors(e,r=3){const s=e.length-r,o=new Float32Array(2*s);if(r===3)for(let i=0;i<s;i+=r)o[2*i]=e[i],o[2*i+1]=e[i+1],o[2*i+2]=e[i+2],o[2*i+3]=e[i+3],o[2*i+4]=e[i+4],o[2*i+5]=e[i+5];else for(let i=0;i<s;i+=r)o[2*i]=e[i],o[2*i+1]=e[i+1],o[2*i+2]=e[i+2],o[2*i+3]=e[i+3],o[2*i+4]=e[i+4],o[2*i+5]=e[i+5],o[2*i+6]=e[i+6],o[2*i+7]=e[i+7];return super.setColors(o,r),this}fromLine(e){const r=e.geometry;return this.setPositions(r.attributes.position.array),this}}class dt extends dn{constructor(e){super({type:"LineMaterial",uniforms:ft.clone(ft.merge([pt.common,pt.fog,{worldUnits:{value:1},linewidth:{value:1},resolution:{value:new Wt(1,1)},dashOffset:{value:0},dashScale:{value:1},dashSize:{value:1},gapSize:{value:1}}])),vertexShader:`
				#include <common>
				#include <fog_pars_vertex>
				#include <logdepthbuf_pars_vertex>
				#include <clipping_planes_pars_vertex>

				uniform float linewidth;
				uniform vec2 resolution;

				attribute vec3 instanceStart;
				attribute vec3 instanceEnd;

				#ifdef USE_COLOR
					#ifdef USE_LINE_COLOR_ALPHA
						varying vec4 vLineColor;
						attribute vec4 instanceColorStart;
						attribute vec4 instanceColorEnd;
					#else
						varying vec3 vLineColor;
						attribute vec3 instanceColorStart;
						attribute vec3 instanceColorEnd;
					#endif
				#endif

				#ifdef WORLD_UNITS

					varying vec4 worldPos;
					varying vec3 worldStart;
					varying vec3 worldEnd;

					#ifdef USE_DASH

						varying vec2 vUv;

					#endif

				#else

					varying vec2 vUv;

				#endif

				#ifdef USE_DASH

					uniform float dashScale;
					attribute float instanceDistanceStart;
					attribute float instanceDistanceEnd;
					varying float vLineDistance;

				#endif

				void trimSegment( const in vec4 start, inout vec4 end ) {

					// trim end segment so it terminates between the camera plane and the near plane

					// conservative estimate of the near plane
					float a = projectionMatrix[ 2 ][ 2 ]; // 3nd entry in 3th column
					float b = projectionMatrix[ 3 ][ 2 ]; // 3nd entry in 4th column
					float nearEstimate = - 0.5 * b / a;

					float alpha = ( nearEstimate - start.z ) / ( end.z - start.z );

					end.xyz = mix( start.xyz, end.xyz, alpha );

				}

				void main() {

					#ifdef USE_COLOR

						vLineColor = ( position.y < 0.5 ) ? instanceColorStart : instanceColorEnd;

					#endif

					#ifdef USE_DASH

						vLineDistance = ( position.y < 0.5 ) ? dashScale * instanceDistanceStart : dashScale * instanceDistanceEnd;
						vUv = uv;

					#endif

					float aspect = resolution.x / resolution.y;

					// camera space
					vec4 start = modelViewMatrix * vec4( instanceStart, 1.0 );
					vec4 end = modelViewMatrix * vec4( instanceEnd, 1.0 );

					#ifdef WORLD_UNITS

						worldStart = start.xyz;
						worldEnd = end.xyz;

					#else

						vUv = uv;

					#endif

					// special case for perspective projection, and segments that terminate either in, or behind, the camera plane
					// clearly the gpu firmware has a way of addressing this issue when projecting into ndc space
					// but we need to perform ndc-space calculations in the shader, so we must address this issue directly
					// perhaps there is a more elegant solution -- WestLangley

					bool perspective = ( projectionMatrix[ 2 ][ 3 ] == - 1.0 ); // 4th entry in the 3rd column

					if ( perspective ) {

						if ( start.z < 0.0 && end.z >= 0.0 ) {

							trimSegment( start, end );

						} else if ( end.z < 0.0 && start.z >= 0.0 ) {

							trimSegment( end, start );

						}

					}

					// clip space
					vec4 clipStart = projectionMatrix * start;
					vec4 clipEnd = projectionMatrix * end;

					// ndc space
					vec3 ndcStart = clipStart.xyz / clipStart.w;
					vec3 ndcEnd = clipEnd.xyz / clipEnd.w;

					// direction
					vec2 dir = ndcEnd.xy - ndcStart.xy;

					// account for clip-space aspect ratio
					dir.x *= aspect;
					dir = normalize( dir );

					#ifdef WORLD_UNITS

						// get the offset direction as perpendicular to the view vector
						vec3 worldDir = normalize( end.xyz - start.xyz );
						vec3 offset;
						if ( position.y < 0.5 ) {

							offset = normalize( cross( start.xyz, worldDir ) );

						} else {

							offset = normalize( cross( end.xyz, worldDir ) );

						}

						// sign flip
						if ( position.x < 0.0 ) offset *= - 1.0;

						float forwardOffset = dot( worldDir, vec3( 0.0, 0.0, 1.0 ) );

						// don't extend the line if we're rendering dashes because we
						// won't be rendering the endcaps
						#ifndef USE_DASH

							// extend the line bounds to encompass  endcaps
							start.xyz += - worldDir * linewidth * 0.5;
							end.xyz += worldDir * linewidth * 0.5;

							// shift the position of the quad so it hugs the forward edge of the line
							offset.xy -= dir * forwardOffset;
							offset.z += 0.5;

						#endif

						// endcaps
						if ( position.y > 1.0 || position.y < 0.0 ) {

							offset.xy += dir * 2.0 * forwardOffset;

						}

						// adjust for linewidth
						offset *= linewidth * 0.5;

						// set the world position
						worldPos = ( position.y < 0.5 ) ? start : end;
						worldPos.xyz += offset;

						// project the worldpos
						vec4 clip = projectionMatrix * worldPos;

						// shift the depth of the projected points so the line
						// segments overlap neatly
						vec3 clipPose = ( position.y < 0.5 ) ? ndcStart : ndcEnd;
						clip.z = clipPose.z * clip.w;

					#else

						vec2 offset = vec2( dir.y, - dir.x );
						// undo aspect ratio adjustment
						dir.x /= aspect;
						offset.x /= aspect;

						// sign flip
						if ( position.x < 0.0 ) offset *= - 1.0;

						// endcaps
						if ( position.y < 0.0 ) {

							offset += - dir;

						} else if ( position.y > 1.0 ) {

							offset += dir;

						}

						// adjust for linewidth
						offset *= linewidth;

						// adjust for clip-space to screen-space conversion // maybe resolution should be based on viewport ...
						offset /= resolution.y;

						// select end
						vec4 clip = ( position.y < 0.5 ) ? clipStart : clipEnd;

						// back to clip space
						offset *= clip.w;

						clip.xy += offset;

					#endif

					gl_Position = clip;

					vec4 mvPosition = ( position.y < 0.5 ) ? start : end; // this is an approximation

					#include <logdepthbuf_vertex>
					#include <clipping_planes_vertex>
					#include <fog_vertex>

				}
			`,fragmentShader:`
				uniform vec3 diffuse;
				uniform float opacity;
				uniform float linewidth;

				#ifdef USE_DASH

					uniform float dashOffset;
					uniform float dashSize;
					uniform float gapSize;

				#endif

				varying float vLineDistance;

				#ifdef WORLD_UNITS

					varying vec4 worldPos;
					varying vec3 worldStart;
					varying vec3 worldEnd;

					#ifdef USE_DASH

						varying vec2 vUv;

					#endif

				#else

					varying vec2 vUv;

				#endif

				#include <common>
				#include <fog_pars_fragment>
				#include <logdepthbuf_pars_fragment>
				#include <clipping_planes_pars_fragment>

				#ifdef USE_COLOR
					#ifdef USE_LINE_COLOR_ALPHA
						varying vec4 vLineColor;
					#else
						varying vec3 vLineColor;
					#endif
				#endif

				vec2 closestLineToLine(vec3 p1, vec3 p2, vec3 p3, vec3 p4) {

					float mua;
					float mub;

					vec3 p13 = p1 - p3;
					vec3 p43 = p4 - p3;

					vec3 p21 = p2 - p1;

					float d1343 = dot( p13, p43 );
					float d4321 = dot( p43, p21 );
					float d1321 = dot( p13, p21 );
					float d4343 = dot( p43, p43 );
					float d2121 = dot( p21, p21 );

					float denom = d2121 * d4343 - d4321 * d4321;

					float numer = d1343 * d4321 - d1321 * d4343;

					mua = numer / denom;
					mua = clamp( mua, 0.0, 1.0 );
					mub = ( d1343 + d4321 * ( mua ) ) / d4343;
					mub = clamp( mub, 0.0, 1.0 );

					return vec2( mua, mub );

				}

				void main() {

					#include <clipping_planes_fragment>

					#ifdef USE_DASH

						if ( vUv.y < - 1.0 || vUv.y > 1.0 ) discard; // discard endcaps

						if ( mod( vLineDistance + dashOffset, dashSize + gapSize ) > dashSize ) discard; // todo - FIX

					#endif

					float alpha = opacity;

					#ifdef WORLD_UNITS

						// Find the closest points on the view ray and the line segment
						vec3 rayEnd = normalize( worldPos.xyz ) * 1e5;
						vec3 lineDir = worldEnd - worldStart;
						vec2 params = closestLineToLine( worldStart, worldEnd, vec3( 0.0, 0.0, 0.0 ), rayEnd );

						vec3 p1 = worldStart + lineDir * params.x;
						vec3 p2 = rayEnd * params.y;
						vec3 delta = p1 - p2;
						float len = length( delta );
						float norm = len / linewidth;

						#ifndef USE_DASH

							#ifdef USE_ALPHA_TO_COVERAGE

								float dnorm = fwidth( norm );
								alpha = 1.0 - smoothstep( 0.5 - dnorm, 0.5 + dnorm, norm );

							#else

								if ( norm > 0.5 ) {

									discard;

								}

							#endif

						#endif

					#else

						#ifdef USE_ALPHA_TO_COVERAGE

							// artifacts appear on some hardware if a derivative is taken within a conditional
							float a = vUv.x;
							float b = ( vUv.y > 0.0 ) ? vUv.y - 1.0 : vUv.y + 1.0;
							float len2 = a * a + b * b;
							float dlen = fwidth( len2 );

							if ( abs( vUv.y ) > 1.0 ) {

								alpha = 1.0 - smoothstep( 1.0 - dlen, 1.0 + dlen, len2 );

							}

						#else

							if ( abs( vUv.y ) > 1.0 ) {

								float a = vUv.x;
								float b = ( vUv.y > 0.0 ) ? vUv.y - 1.0 : vUv.y + 1.0;
								float len2 = a * a + b * b;

								if ( len2 > 1.0 ) discard;

							}

						#endif

					#endif

					vec4 diffuseColor = vec4( diffuse, alpha );
					#ifdef USE_COLOR
						#ifdef USE_LINE_COLOR_ALPHA
							diffuseColor *= vLineColor;
						#else
							diffuseColor.rgb *= vLineColor;
						#endif
					#endif

					#include <logdepthbuf_fragment>

					gl_FragColor = diffuseColor;

					#include <tonemapping_fragment>
					#include <${Yt>=154?"colorspace_fragment":"encodings_fragment"}>
					#include <fog_fragment>
					#include <premultiplied_alpha_fragment>

				}
			`,clipping:!0}),this.isLineMaterial=!0,this.onBeforeCompile=function(){this.transparent?this.defines.USE_LINE_COLOR_ALPHA="1":delete this.defines.USE_LINE_COLOR_ALPHA},Object.defineProperties(this,{color:{enumerable:!0,get:function(){return this.uniforms.diffuse.value},set:function(r){this.uniforms.diffuse.value=r}},worldUnits:{enumerable:!0,get:function(){return"WORLD_UNITS"in this.defines},set:function(r){r===!0?this.defines.WORLD_UNITS="":delete this.defines.WORLD_UNITS}},linewidth:{enumerable:!0,get:function(){return this.uniforms.linewidth.value},set:function(r){this.uniforms.linewidth.value=r}},dashed:{enumerable:!0,get:function(){return"USE_DASH"in this.defines},set(r){!!r!="USE_DASH"in this.defines&&(this.needsUpdate=!0),r===!0?this.defines.USE_DASH="":delete this.defines.USE_DASH}},dashScale:{enumerable:!0,get:function(){return this.uniforms.dashScale.value},set:function(r){this.uniforms.dashScale.value=r}},dashSize:{enumerable:!0,get:function(){return this.uniforms.dashSize.value},set:function(r){this.uniforms.dashSize.value=r}},dashOffset:{enumerable:!0,get:function(){return this.uniforms.dashOffset.value},set:function(r){this.uniforms.dashOffset.value=r}},gapSize:{enumerable:!0,get:function(){return this.uniforms.gapSize.value},set:function(r){this.uniforms.gapSize.value=r}},opacity:{enumerable:!0,get:function(){return this.uniforms.opacity.value},set:function(r){this.uniforms.opacity.value=r}},resolution:{enumerable:!0,get:function(){return this.uniforms.resolution.value},set:function(r){this.uniforms.resolution.value.copy(r)}},alphaToCoverage:{enumerable:!0,get:function(){return"USE_ALPHA_TO_COVERAGE"in this.defines},set:function(r){!!r!="USE_ALPHA_TO_COVERAGE"in this.defines&&(this.needsUpdate=!0),r===!0?(this.defines.USE_ALPHA_TO_COVERAGE="",this.extensions.derivatives=!0):(delete this.defines.USE_ALPHA_TO_COVERAGE,this.extensions.derivatives=!1)}}}),this.setValues(e)}}const Ie=new ye,gt=new w,yt=new w,X=new ye,K=new ye,ne=new ye,$e=new w,He=new $,Y=new mn,vt=new w,Pe=new Me,_e=new Nt,re=new ye;let oe,de;function wt(n,e,r){return re.set(0,0,-e,1).applyMatrix4(n.projectionMatrix),re.multiplyScalar(1/re.w),re.x=de/r.width,re.y=de/r.height,re.applyMatrix4(n.projectionMatrixInverse),re.multiplyScalar(1/re.w),Math.abs(Math.max(re.x,re.y))}function Sn(n,e){const r=n.matrixWorld,s=n.geometry,o=s.attributes.instanceStart,i=s.attributes.instanceEnd,c=Math.min(s.instanceCount,o.count);for(let d=0,p=c;d<p;d++){Y.start.fromBufferAttribute(o,d),Y.end.fromBufferAttribute(i,d),Y.applyMatrix4(r);const x=new w,u=new w;oe.distanceSqToSegment(Y.start,Y.end,u,x),u.distanceTo(x)<de*.5&&e.push({point:u,pointOnLine:x,distance:oe.origin.distanceTo(u),object:n,face:null,faceIndex:d,uv:null,[Zt]:null})}}function jn(n,e,r){const s=e.projectionMatrix,i=n.material.resolution,c=n.matrixWorld,d=n.geometry,p=d.attributes.instanceStart,x=d.attributes.instanceEnd,u=Math.min(d.instanceCount,p.count),l=-e.near;oe.at(1,ne),ne.w=1,ne.applyMatrix4(e.matrixWorldInverse),ne.applyMatrix4(s),ne.multiplyScalar(1/ne.w),ne.x*=i.x/2,ne.y*=i.y/2,ne.z=0,$e.copy(ne),He.multiplyMatrices(e.matrixWorldInverse,c);for(let v=0,b=u;v<b;v++){if(X.fromBufferAttribute(p,v),K.fromBufferAttribute(x,v),X.w=1,K.w=1,X.applyMatrix4(He),K.applyMatrix4(He),X.z>l&&K.z>l)continue;if(X.z>l){const f=X.z-K.z,E=(X.z-l)/f;X.lerp(K,E)}else if(K.z>l){const f=K.z-X.z,E=(K.z-l)/f;K.lerp(X,E)}X.applyMatrix4(s),K.applyMatrix4(s),X.multiplyScalar(1/X.w),K.multiplyScalar(1/K.w),X.x*=i.x/2,X.y*=i.y/2,K.x*=i.x/2,K.y*=i.y/2,Y.start.copy(X),Y.start.z=0,Y.end.copy(K),Y.end.z=0;const C=Y.closestPointToPointParameter($e,!0);Y.at(C,vt);const F=It.lerp(X.z,K.z,C),_=F>=-1&&F<=1,D=$e.distanceTo(vt)<de*.5;if(_&&D){Y.start.fromBufferAttribute(p,v),Y.end.fromBufferAttribute(x,v),Y.start.applyMatrix4(c),Y.end.applyMatrix4(c);const f=new w,E=new w;oe.distanceSqToSegment(Y.start,Y.end,E,f),r.push({point:E,pointOnLine:f,distance:oe.origin.distanceTo(E),object:n,face:null,faceIndex:v,uv:null,[Zt]:null})}}}class en extends un{constructor(e=new ct,r=new dt({color:Math.random()*16777215})){super(e,r),this.isLineSegments2=!0,this.type="LineSegments2"}computeLineDistances(){const e=this.geometry,r=e.attributes.instanceStart,s=e.attributes.instanceEnd,o=new Float32Array(2*r.count);for(let c=0,d=0,p=r.count;c<p;c++,d+=2)gt.fromBufferAttribute(r,c),yt.fromBufferAttribute(s,c),o[d]=d===0?0:o[d-1],o[d+1]=o[d]+gt.distanceTo(yt);const i=new nt(o,2,1);return e.setAttribute("instanceDistanceStart",new ge(i,1,0)),e.setAttribute("instanceDistanceEnd",new ge(i,1,1)),this}raycast(e,r){const s=this.material.worldUnits,o=e.camera;o===null&&!s&&console.error('LineSegments2: "Raycaster.camera" needs to be set in order to raycast against LineSegments2 while worldUnits is set to false.');const i=e.params.Line2!==void 0&&e.params.Line2.threshold||0;oe=e.ray;const c=this.matrixWorld,d=this.geometry,p=this.material;de=p.linewidth+i,d.boundingSphere===null&&d.computeBoundingSphere(),_e.copy(d.boundingSphere).applyMatrix4(c);let x;if(s)x=de*.5;else{const l=Math.max(o.near,_e.distanceToPoint(oe.origin));x=wt(o,l,p.resolution)}if(_e.radius+=x,oe.intersectsSphere(_e)===!1)return;d.boundingBox===null&&d.computeBoundingBox(),Pe.copy(d.boundingBox).applyMatrix4(c);let u;if(s)u=de*.5;else{const l=Math.max(o.near,Pe.distanceToPoint(oe.origin));u=wt(o,l,p.resolution)}Pe.expandByScalar(u),oe.intersectsBox(Pe)!==!1&&(s?Sn(this,r):jn(this,o,r))}onBeforeRender(e){const r=this.material.uniforms;r&&r.resolution&&(e.getViewport(Ie),this.material.uniforms.resolution.value.set(Ie.z,Ie.w))}}class Mn extends en{constructor(e=new Jt,r=new dt({color:Math.random()*16777215})){super(e,r),this.isLine2=!0,this.type="Line2"}}const En=new w,kn=new w,Pn=new w,_n=(n,e,r)=>{const s=r.width/2,o=r.height/2;e.updateMatrixWorld(!1);const i=n.project(e);return i.x=i.x*s+s,i.y=-(i.y*o)+o,i},Cn=(n,e,r,s=1)=>{const o=En.set(n.x/r.width*2-1,-(n.y/r.height)*2+1,s);return o.unproject(e),o},tn=(n,e,r,s)=>{const o=_n(Pn.copy(n),r,s);let i=0;for(let c=0;c<2;++c){const d=kn.copy(o).setComponent(c,o.getComponent(c)+e),p=Cn(d,r,s,d.z);i=Math.max(i,n.distanceTo(p))}return i},Be=a.forwardRef(function({points:e,color:r=16777215,vertexColors:s,linewidth:o,lineWidth:i,segments:c,dashed:d,...p},x){var u,l;const v=le(_=>_.size),b=a.useMemo(()=>c?new en:new Mn,[c]),[S]=a.useState(()=>new dt),C=(s==null||(u=s[0])==null?void 0:u.length)===4?4:3,F=a.useMemo(()=>{const _=c?new ct:new Jt,D=e.map(f=>{const E=Array.isArray(f);return f instanceof w||f instanceof ye?[f.x,f.y,f.z]:f instanceof Wt?[f.x,f.y,0]:E&&f.length===3?[f[0],f[1],f[2]]:E&&f.length===2?[f[0],f[1],0]:f});if(_.setPositions(D.flat()),s){r=16777215;const f=s.map(E=>E instanceof fn?E.toArray():E);_.setColors(f.flat(),C)}return _},[e,c,s,C]);return a.useLayoutEffect(()=>{b.computeLineDistances()},[e,b]),a.useLayoutEffect(()=>{d?S.defines.USE_DASH="":delete S.defines.USE_DASH,S.needsUpdate=!0},[d,S]),a.useEffect(()=>()=>{F.dispose(),S.dispose()},[F]),a.createElement("primitive",rt({object:b,ref:x},p),a.createElement("primitive",{object:F,attach:"geometry"}),a.createElement("primitive",rt({object:S,attach:"material",color:r,vertexColors:!!s,resolution:[v.width,v.height],linewidth:(l=o??i)!==null&&l!==void 0?l:1,dashed:d,transparent:C===4},p)))}),Ee=a.createContext(null),Ce=new w,bt=new w,Ln=(n,e,r,s)=>{const o=e.dot(e),i=e.dot(n)-e.dot(r),c=e.dot(s);return c===0?-i/o:(Ce.copy(s).multiplyScalar(o/c).sub(e),bt.copy(s).multiplyScalar(i/c).add(r).sub(n),-Ce.dot(bt)/Ce.dot(Ce))},Dn=new w(0,1,0),St=new $,Ge=({direction:n,axis:e})=>{const{translation:r,translationLimits:s,annotations:o,annotationsClass:i,depthTest:c,scale:d,lineWidth:p,fixed:x,axisColors:u,hoveredColor:l,opacity:v,onDragStart:b,onDrag:S,onDragEnd:C,userData:F}=a.useContext(Ee),_=le(j=>j.controls),D=a.useRef(null),f=a.useRef(null),E=a.useRef(null),z=a.useRef(0),[O,W]=a.useState(!1),H=a.useCallback(j=>{o&&(D.current.innerText=`${r.current[e].toFixed(2)}`,D.current.style.display="block"),j.stopPropagation();const g=new $().extractRotation(f.current.matrixWorld),M=j.point.clone(),m=new w().setFromMatrixPosition(f.current.matrixWorld),y=n.clone().applyMatrix4(g).normalize();E.current={clickPoint:M,dir:y},z.current=r.current[e],b({component:"Arrow",axis:e,origin:m,directions:[y]}),_&&(_.enabled=!1),j.target.setPointerCapture(j.pointerId)},[o,n,_,b,r,e]),q=a.useCallback(j=>{if(j.stopPropagation(),O||W(!0),E.current){const{clickPoint:g,dir:M}=E.current,[m,y]=(s==null?void 0:s[e])||[void 0,void 0];let k=Ln(g,M,j.ray.origin,j.ray.direction);m!==void 0&&(k=Math.max(k,m-z.current)),y!==void 0&&(k=Math.min(k,y-z.current)),r.current[e]=z.current+k,o&&(D.current.innerText=`${r.current[e].toFixed(2)}`),St.makeTranslation(M.x*k,M.y*k,M.z*k),S(St)}},[o,S,O,r,s,e]),h=a.useCallback(j=>{o&&(D.current.style.display="none"),j.stopPropagation(),E.current=null,C(),_&&(_.enabled=!0),j.target.releasePointerCapture(j.pointerId)},[o,_,C]),P=a.useCallback(j=>{j.stopPropagation(),W(!1)},[]),{cylinderLength:A,coneWidth:U,coneLength:B,matrixL:N}=a.useMemo(()=>{const j=x?p/d*1.6:d/20,g=x?.2:d/5,M=x?1-g:d-g,m=new at().setFromUnitVectors(Dn,n.clone().normalize()),y=new $().makeRotationFromQuaternion(m);return{cylinderLength:M,coneWidth:j,coneLength:g,matrixL:y}},[n,d,p,x]),L=O?l:u[e];return a.createElement("group",{ref:f},a.createElement("group",{matrix:N,matrixAutoUpdate:!1,onPointerDown:H,onPointerMove:q,onPointerUp:h,onPointerOut:P},o&&a.createElement(Oe,{position:[0,-B,0]},a.createElement("div",{style:{display:"none",background:"#151520",color:"white",padding:"6px 8px",borderRadius:7,whiteSpace:"nowrap"},className:i,ref:D})),a.createElement("mesh",{visible:!1,position:[0,(A+B)/2,0],userData:F},a.createElement("cylinderGeometry",{args:[U*1.4,U*1.4,A+B,8,1]})),a.createElement(Be,{transparent:!0,raycast:()=>null,depthTest:c,points:[0,0,0,0,A,0],lineWidth:p,side:Ue,color:L,opacity:v,polygonOffset:!0,renderOrder:1,polygonOffsetFactor:-10,fog:!1}),a.createElement("mesh",{raycast:()=>null,position:[0,A+B/2,0],renderOrder:500},a.createElement("coneGeometry",{args:[U,B,24,1]}),a.createElement("meshBasicMaterial",{transparent:!0,depthTest:c,color:L,opacity:v,polygonOffset:!0,polygonOffsetFactor:-10,fog:!1}))))},qe=new w,Ve=new w,Xe=n=>n*180/Math.PI,Tn=n=>n*Math.PI/180,Fn=(n,e,r,s,o)=>{qe.copy(n).sub(r),Ve.copy(e).sub(r);const i=s.dot(s),c=o.dot(o),d=qe.dot(s)/i,p=qe.dot(o)/c,x=Ve.dot(s)/i,u=Ve.dot(o)/c,l=Math.atan2(p,d);return Math.atan2(u,x)-l},zn=(n,e)=>{let r=Math.floor(n/e);return r=r<0?r+1:r,n-r*e},jt=n=>{let e=zn(n,2*Math.PI);return Math.abs(e)<1e-6?0:(e<0&&(e+=2*Math.PI),e)},Le=new $,Mt=new w,De=new Ht,Ke=new w,Qe=({dir1:n,dir2:e,axis:r})=>{const{rotationLimits:s,annotations:o,annotationsClass:i,depthTest:c,scale:d,lineWidth:p,fixed:x,axisColors:u,hoveredColor:l,opacity:v,onDragStart:b,onDrag:S,onDragEnd:C,userData:F}=a.useContext(Ee),_=le(L=>L.controls),D=a.useRef(null),f=a.useRef(null),E=a.useRef(0),z=a.useRef(0),O=a.useRef(null),[W,H]=a.useState(!1),q=a.useCallback(L=>{o&&(D.current.innerText=`${Xe(z.current).toFixed(0)}º`,D.current.style.display="block"),L.stopPropagation();const j=L.point.clone(),g=new w().setFromMatrixPosition(f.current.matrixWorld),M=new w().setFromMatrixColumn(f.current.matrixWorld,0).normalize(),m=new w().setFromMatrixColumn(f.current.matrixWorld,1).normalize(),y=new w().setFromMatrixColumn(f.current.matrixWorld,2).normalize(),k=new $t().setFromNormalAndCoplanarPoint(y,g);O.current={clickPoint:j,origin:g,e1:M,e2:m,normal:y,plane:k},b({component:"Rotator",axis:r,origin:g,directions:[M,m,y]}),_&&(_.enabled=!1),L.target.setPointerCapture(L.pointerId)},[o,_,b,r]),h=a.useCallback(L=>{if(L.stopPropagation(),W||H(!0),O.current){const{clickPoint:j,origin:g,e1:M,e2:m,normal:y,plane:k}=O.current,[R,G]=(s==null?void 0:s[r])||[void 0,void 0];De.copy(L.ray),De.intersectPlane(k,Ke),De.direction.negate(),De.intersectPlane(k,Ke);let T=Fn(j,Ke,g,M,m),ee=Xe(T);L.shiftKey&&(ee=Math.round(ee/10)*10,T=Tn(ee)),R!==void 0&&G!==void 0&&G-R<2*Math.PI?(T=jt(T),T=T>Math.PI?T-2*Math.PI:T,T=It.clamp(T,R-E.current,G-E.current),z.current=E.current+T):(z.current=jt(E.current+T),z.current=z.current>Math.PI?z.current-2*Math.PI:z.current),o&&(ee=Xe(z.current),D.current.innerText=`${ee.toFixed(0)}º`),Le.makeRotationAxis(y,T),Mt.copy(g).applyMatrix4(Le).sub(g).negate(),Le.setPosition(Mt),S(Le)}},[o,S,W,s,r]),P=a.useCallback(L=>{o&&(D.current.style.display="none"),L.stopPropagation(),E.current=z.current,O.current=null,C(),_&&(_.enabled=!0),L.target.releasePointerCapture(L.pointerId)},[o,_,C]),A=a.useCallback(L=>{L.stopPropagation(),H(!1)},[]),U=a.useMemo(()=>{const L=n.clone().normalize(),j=e.clone().normalize();return new $().makeBasis(L,j,L.clone().cross(j))},[n,e]),B=x?.65:d*.65,N=a.useMemo(()=>{const j=[];for(let g=0;g<=32;g++){const M=g*(Math.PI/2)/32;j.push(new w(Math.cos(M)*B,Math.sin(M)*B,0))}return j},[B]);return a.createElement("group",{ref:f,onPointerDown:q,onPointerMove:h,onPointerUp:P,onPointerOut:A,matrix:U,matrixAutoUpdate:!1},o&&a.createElement(Oe,{position:[B,B,0]},a.createElement("div",{style:{display:"none",background:"#151520",color:"white",padding:"6px 8px",borderRadius:7,whiteSpace:"nowrap"},className:i,ref:D})),a.createElement(Be,{points:N,lineWidth:p*4,visible:!1,userData:F}),a.createElement(Be,{transparent:!0,raycast:()=>null,depthTest:c,points:N,lineWidth:p,side:Ue,color:W?l:u[r],opacity:v,polygonOffset:!0,polygonOffsetFactor:-10,fog:!1}))},Rn=(n,e,r)=>{const s=Math.abs(n.x)>=Math.abs(n.y)&&Math.abs(n.x)>=Math.abs(n.z)?0:Math.abs(n.y)>=Math.abs(n.x)&&Math.abs(n.y)>=Math.abs(n.z)?1:2,o=[0,1,2].sort((S,C)=>Math.abs(e.getComponent(C))-Math.abs(e.getComponent(S))),i=s===o[0]?o[1]:o[0],c=n.getComponent(s),d=n.getComponent(i),p=e.getComponent(s),x=e.getComponent(i),u=r.getComponent(s),v=(r.getComponent(i)-u*(d/c))/(x-p*(d/c));return[(u-v*p)/c,v]},Te=new Ht,Fe=new w,Et=new $,Ye=({dir1:n,dir2:e,axis:r})=>{const{translation:s,translationLimits:o,annotations:i,annotationsClass:c,depthTest:d,scale:p,lineWidth:x,fixed:u,axisColors:l,hoveredColor:v,opacity:b,onDragStart:S,onDrag:C,onDragEnd:F,userData:_}=a.useContext(Ee),D=le(M=>M.controls),f=a.useRef(null),E=a.useRef(null),z=a.useRef(null),O=a.useRef(0),W=a.useRef(0),[H,q]=a.useState(!1),h=a.useCallback(M=>{i&&(f.current.innerText=`${s.current[(r+1)%3].toFixed(2)}, ${s.current[(r+2)%3].toFixed(2)}`,f.current.style.display="block"),M.stopPropagation();const m=M.point.clone(),y=new w().setFromMatrixPosition(E.current.matrixWorld),k=new w().setFromMatrixColumn(E.current.matrixWorld,0).normalize(),R=new w().setFromMatrixColumn(E.current.matrixWorld,1).normalize(),G=new w().setFromMatrixColumn(E.current.matrixWorld,2).normalize(),T=new $t().setFromNormalAndCoplanarPoint(G,y);z.current={clickPoint:m,e1:k,e2:R,plane:T},O.current=s.current[(r+1)%3],W.current=s.current[(r+2)%3],S({component:"Slider",axis:r,origin:y,directions:[k,R,G]}),D&&(D.enabled=!1),M.target.setPointerCapture(M.pointerId)},[i,D,S,r]),P=a.useCallback(M=>{if(M.stopPropagation(),H||q(!0),z.current){const{clickPoint:m,e1:y,e2:k,plane:R}=z.current,[G,T]=(o==null?void 0:o[(r+1)%3])||[void 0,void 0],[ee,ue]=(o==null?void 0:o[(r+2)%3])||[void 0,void 0];Te.copy(M.ray),Te.intersectPlane(R,Fe),Te.direction.negate(),Te.intersectPlane(R,Fe),Fe.sub(m);let[te,V]=Rn(y,k,Fe);G!==void 0&&(te=Math.max(te,G-O.current)),T!==void 0&&(te=Math.min(te,T-O.current)),ee!==void 0&&(V=Math.max(V,ee-W.current)),ue!==void 0&&(V=Math.min(V,ue-W.current)),s.current[(r+1)%3]=O.current+te,s.current[(r+2)%3]=W.current+V,i&&(f.current.innerText=`${s.current[(r+1)%3].toFixed(2)}, ${s.current[(r+2)%3].toFixed(2)}`),Et.makeTranslation(te*y.x+V*k.x,te*y.y+V*k.y,te*y.z+V*k.z),C(Et)}},[i,C,H,s,o,r]),A=a.useCallback(M=>{i&&(f.current.style.display="none"),M.stopPropagation(),z.current=null,F(),D&&(D.enabled=!0),M.target.releasePointerCapture(M.pointerId)},[i,D,F]),U=a.useCallback(M=>{M.stopPropagation(),q(!1)},[]),B=a.useMemo(()=>{const M=n.clone().normalize(),m=e.clone().normalize();return new $().makeBasis(M,m,M.clone().cross(m))},[n,e]),N=u?1/7:p/7,L=u?.225:p*.225,j=H?v:l[r],g=a.useMemo(()=>[new w(0,0,0),new w(0,L,0),new w(L,L,0),new w(L,0,0),new w(0,0,0)],[L]);return a.createElement("group",{ref:E,matrix:B,matrixAutoUpdate:!1},i&&a.createElement(Oe,{position:[0,0,0]},a.createElement("div",{style:{display:"none",background:"#151520",color:"white",padding:"6px 8px",borderRadius:7,whiteSpace:"nowrap"},className:c,ref:f})),a.createElement("group",{position:[N*1.7,N*1.7,0]},a.createElement("mesh",{visible:!0,onPointerDown:h,onPointerMove:P,onPointerUp:A,onPointerOut:U,scale:L,userData:_},a.createElement("planeGeometry",null),a.createElement("meshBasicMaterial",{transparent:!0,depthTest:d,color:j,polygonOffset:!0,polygonOffsetFactor:-10,side:Ue,fog:!1})),a.createElement(Be,{position:[-L/2,-L/2,0],transparent:!0,depthTest:d,points:g,lineWidth:x,color:j,opacity:b,polygonOffset:!0,polygonOffsetFactor:-10,userData:_,fog:!1})))},be=new w,kt=new w,Bn=(n,e,r,s)=>{const o=e.dot(e),i=e.dot(n)-e.dot(r),c=e.dot(s);return c===0?-i/o:(be.copy(s).multiplyScalar(o/c).sub(e),kt.copy(s).multiplyScalar(i/c).add(r).sub(n),-be.dot(kt)/be.dot(be))},An=new w(0,1,0),ve=new w,Pt=new $,Ze=({direction:n,axis:e})=>{const{scaleLimits:r,annotations:s,annotationsClass:o,depthTest:i,scale:c,lineWidth:d,fixed:p,axisColors:x,hoveredColor:u,opacity:l,onDragStart:v,onDrag:b,onDragEnd:S,userData:C}=a.useContext(Ee),F=le(g=>g.size),_=le(g=>g.controls),D=a.useRef(null),f=a.useRef(null),E=a.useRef(null),z=a.useRef(1),O=a.useRef(1),W=a.useRef(null),[H,q]=a.useState(!1),h=p?1.2:1.2*c,P=a.useCallback(g=>{s&&(D.current.innerText=`${O.current.toFixed(2)}`,D.current.style.display="block"),g.stopPropagation();const M=new $().extractRotation(f.current.matrixWorld),m=g.point.clone(),y=new w().setFromMatrixPosition(f.current.matrixWorld),k=n.clone().applyMatrix4(M).normalize(),R=f.current.matrixWorld.clone(),G=R.clone().invert(),T=p?1/tn(f.current.getWorldPosition(be),c,g.camera,F):1;W.current={clickPoint:m,dir:k,mPLG:R,mPLGInv:G,offsetMultiplier:T},v({component:"Sphere",axis:e,origin:y,directions:[k]}),_&&(_.enabled=!1),g.target.setPointerCapture(g.pointerId)},[s,_,n,v,e,p,c,F]),A=a.useCallback(g=>{if(g.stopPropagation(),H||q(!0),W.current){const{clickPoint:M,dir:m,mPLG:y,mPLGInv:k,offsetMultiplier:R}=W.current,[G,T]=(r==null?void 0:r[e])||[1e-5,void 0],ue=Bn(M,m,g.ray.origin,g.ray.direction)*R,te=p?ue:ue/c;let V=Math.pow(2,te*.2);g.shiftKey&&(V=Math.round(V*10)/10),V=Math.max(V,G/z.current),T!==void 0&&(V=Math.min(V,T/z.current)),O.current=z.current*V,E.current.position.set(0,h+ue,0),s&&(D.current.innerText=`${O.current.toFixed(2)}`),ve.set(1,1,1),ve.setComponent(e,V),Pt.makeScale(ve.x,ve.y,ve.z).premultiply(y).multiply(k),b(Pt)}},[s,h,b,H,r,e]),U=a.useCallback(g=>{s&&(D.current.style.display="none"),g.stopPropagation(),z.current=O.current,W.current=null,E.current.position.set(0,h,0),S(),_&&(_.enabled=!0),g.target.releasePointerCapture(g.pointerId)},[s,_,S,h]),B=a.useCallback(g=>{g.stopPropagation(),q(!1)},[]),{radius:N,matrixL:L}=a.useMemo(()=>{const g=p?d/c*1.8:c/22.5,M=new at().setFromUnitVectors(An,n.clone().normalize()),m=new $().makeRotationFromQuaternion(M);return{radius:g,matrixL:m}},[n,c,d,p]),j=H?u:x[e];return a.createElement("group",{ref:f},a.createElement("group",{matrix:L,matrixAutoUpdate:!1,onPointerDown:P,onPointerMove:A,onPointerUp:U,onPointerOut:B},s&&a.createElement(Oe,{position:[0,h/2,0]},a.createElement("div",{style:{display:"none",background:"#151520",color:"white",padding:"6px 8px",borderRadius:7,whiteSpace:"nowrap"},className:o,ref:D})),a.createElement("mesh",{ref:E,position:[0,h,0],renderOrder:500,userData:C},a.createElement("sphereGeometry",{args:[N,12,12]}),a.createElement("meshBasicMaterial",{transparent:!0,depthTest:i,color:j,opacity:l,polygonOffset:!0,polygonOffsetFactor:-10}))))},_t=new $,Ct=new $,Lt=new $,ze=new $,Je=new $,me=new $,Dt=new $,Tt=new $,Ft=new $,fe=new Me,et=new Me,zt=new w,Rt=new w,Bt=new w,At=new w,we=new w,pe=new w(1,0,0),he=new w(0,1,0),xe=new w(0,0,1),On=a.forwardRef(({enabled:n=!0,matrix:e,onDragStart:r,onDrag:s,onDragEnd:o,autoTransform:i=!0,anchor:c,disableAxes:d=!1,disableSliders:p=!1,disableRotations:x=!1,disableScaling:u=!1,activeAxes:l=[!0,!0,!0],offset:v=[0,0,0],rotation:b=[0,0,0],scale:S=1,lineWidth:C=4,fixed:F=!1,translationLimits:_,rotationLimits:D,scaleLimits:f,depthTest:E=!0,axisColors:z=["#ff2060","#20df80","#2080ff"],hoveredColor:O="#ffff40",annotations:W=!1,annotationsClass:H,opacity:q=1,visible:h=!0,userData:P,children:A,...U},B)=>{const N=le(T=>T.invalidate),L=a.useRef(null),j=a.useRef(null),g=a.useRef(null),M=a.useRef(null),m=a.useRef([0,0,0]),y=a.useRef(new w(1,1,1)),k=a.useRef(new w(1,1,1));a.useLayoutEffect(()=>{c&&(M.current.updateWorldMatrix(!0,!0),ze.copy(M.current.matrixWorld).invert(),fe.makeEmpty(),M.current.traverse(T=>{T.geometry&&(T.geometry.boundingBox||T.geometry.computeBoundingBox(),me.copy(T.matrixWorld).premultiply(ze),et.copy(T.geometry.boundingBox),et.applyMatrix4(me),fe.union(et))}),zt.copy(fe.max).add(fe.min).multiplyScalar(.5),Rt.copy(fe.max).sub(fe.min).multiplyScalar(.5),Bt.copy(Rt).multiply(new w(...c)).add(zt),At.set(...v).add(Bt),g.current.position.copy(At),N())});const R=a.useMemo(()=>({onDragStart:T=>{_t.copy(j.current.matrix),Ct.copy(j.current.matrixWorld),r&&r(T),N()},onDrag:T=>{Lt.copy(L.current.matrixWorld),ze.copy(Lt).invert(),Je.copy(Ct).premultiply(T),me.copy(Je).premultiply(ze),Dt.copy(_t).invert(),Tt.copy(me).multiply(Dt),i&&j.current.matrix.copy(me),s&&s(me,Tt,Je,T),N()},onDragEnd:()=>{o&&o(),N()},translation:m,translationLimits:_,rotationLimits:D,axisColors:z,hoveredColor:O,opacity:q,scale:S,lineWidth:C,fixed:F,depthTest:E,userData:P,annotations:W,annotationsClass:H}),[r,s,o,m,_,D,f,E,S,C,F,...z,O,q,P,i,W,H]),G=new w;return Gt(T=>{if(F){const ee=tn(g.current.getWorldPosition(G),S,T.camera,T.size);y.current.setScalar(ee)}e&&e instanceof $&&(j.current.matrix=e),j.current.updateWorldMatrix(!0,!0),Ft.makeRotationFromEuler(g.current.rotation).setPosition(g.current.position).premultiply(j.current.matrixWorld),k.current.setFromMatrixScale(Ft),we.copy(y.current).divide(k.current),(Math.abs(g.current.scale.x-we.x)>1e-4||Math.abs(g.current.scale.y-we.y)>1e-4||Math.abs(g.current.scale.z-we.z)>1e-4)&&(g.current.scale.copy(we),T.invalidate())}),a.useImperativeHandle(B,()=>j.current,[]),a.createElement(Ee.Provider,{value:R},a.createElement("group",{ref:L},a.createElement("group",rt({ref:j,matrix:e,matrixAutoUpdate:!1},U),a.createElement("group",{visible:h,ref:g,position:v,rotation:b},n&&a.createElement(a.Fragment,null,!d&&l[0]&&a.createElement(Ge,{axis:0,direction:pe}),!d&&l[1]&&a.createElement(Ge,{axis:1,direction:he}),!d&&l[2]&&a.createElement(Ge,{axis:2,direction:xe}),!p&&l[0]&&l[1]&&a.createElement(Ye,{axis:2,dir1:pe,dir2:he}),!p&&l[0]&&l[2]&&a.createElement(Ye,{axis:1,dir1:xe,dir2:pe}),!p&&l[2]&&l[1]&&a.createElement(Ye,{axis:0,dir1:he,dir2:xe}),!x&&l[0]&&l[1]&&a.createElement(Qe,{axis:2,dir1:pe,dir2:he}),!x&&l[0]&&l[2]&&a.createElement(Qe,{axis:1,dir1:xe,dir2:pe}),!x&&l[2]&&l[1]&&a.createElement(Qe,{axis:0,dir1:he,dir2:xe}),!u&&l[0]&&a.createElement(Ze,{axis:0,direction:pe}),!u&&l[1]&&a.createElement(Ze,{axis:1,direction:he}),!u&&l[2]&&a.createElement(Ze,{axis:2,direction:xe}))),a.createElement("group",{ref:M},A))))});function er(){const n=ae(),{data:e,error:r}=Ae("pg_slices"),[s,o]=a.useState("input_layernorm"),[i,c]=a.useState(0),d=a.useMemo(()=>{if(!e)return null;const l=e.norms[s];return{vals:Float32Array.from(l.data,v=>1+v),rows:l.shape[0],cols:l.shape[1]}},[e,s]);if(!e||!d)return t.jsx(it,{what:n("real norm weights","gerçek norm ağırlıkları"),error:r});const p=Array.from(tt(e.norms[s],i),l=>1+l),x=Array.from(e.norms.final.data,l=>1+l),u=p.filter(l=>l<0).length;return t.jsxs("div",{children:[t.jsxs("div",{className:"row",style:{justifyContent:"space-between",marginBottom:8},children:[t.jsx(je,{value:s,onChange:o,options:[["input_layernorm","input_layernorm"],["post_attention_layernorm","post_attention_layernorm"]]}),t.jsx("span",{className:"small muted",children:n("color = 1 + w, click a row","renk = 1 + w, satıra tıkla")})]}),t.jsx(on,{values:d.vals,rows:d.rows,cols:d.cols,height:200,diverging:!0,domain:[-4,4],rowLabel:n("layer (0–17)","katman (0–17)"),colLabel:n("dim (0–2047)","boyut (0–2047)"),highlight:{r:i},onCell:l=>c(l),cellLabel:(l,v,b)=>n(`layer ${l}, dim ${v}: 1+w = ${b.toFixed(3)}`,`katman ${l}, boyut ${v}: 1+w = ${b.toFixed(3)}`)}),t.jsxs("div",{className:"grid2",style:{marginTop:14},children:[t.jsxs("div",{children:[t.jsxs("div",{className:"small",children:[t.jsxs("b",{children:[n("Layer","Katman")," ",i]})," · ",s,": ",n("distribution of 1 + w","1 + w dağılımı")]}),t.jsx(ut,{values:p,bins:60,color:"var(--ax-hd)",marks:[{x:1,label:"w=0 → 1"},{x:0,label:"0",color:"var(--bad)"}]})]}),t.jsxs("div",{children:[t.jsxs("div",{className:"small",children:[t.jsx("b",{children:n("Final norm","Son norm")})," (model.norm): ",n("distribution of 1 + w","1 + w dağılımı")]}),t.jsx(ut,{values:x,bins:60,color:"var(--ax-T)",marks:[{x:1,label:"w=0 → 1"}]})]})]}),t.jsxs("div",{className:"row",style:{gap:10,marginTop:10},children:[t.jsx(Se,{label:n("mean scale (1+w)","ortalama ölçek (1+w)"),value:(p.reduce((l,v)=>l+v,0)/p.length).toFixed(2)}),t.jsx(Se,{label:n("largest scale","en büyük ölçek"),value:Math.max(...p).toFixed(2)}),t.jsx(Se,{label:n("1+w < 0 (flips the sign)","1+w < 0 (işaret çeviren)"),value:u,sub:`/ ${p.length} ${n("dims","boyut")}`,tone:u?"warn":void 0})]})]})}function tr(){const n=ae(),{data:e}=Ae("gemma_toy");if(!e)return null;const r=e.rms_order;return t.jsxs("span",{className:"pill bad",children:[r.n_diff," / ",r.n," ",n("elements","eleman")," (",(r.n_diff/r.n*100).toFixed(0),"%) ",n("differ, max diff","farklı, max fark")," ",r.maxdiff]})}function nr({broken:n}){const e=ae(),{data:r}=Ae("gemma_toy");if(!r)return null;const s=r.rms_zero_init_trap;return t.jsxs("div",{className:"row",style:{gap:10},children:[t.jsx(Se,{label:e("with w = 0: difference between correct and buggy","w = 0 iken: doğru ile hatalı arası fark"),value:n?s.zero_w.ignores_weight.toFixed(2):"–",tone:n?"warn":void 0,sub:n?e("the test would PASS, the bug is invisible","test GEÇERDİ, hata görünmez"):""}),t.jsx(Se,{label:e("difference with random w","w rastgele iken fark"),value:n?s.random_w.ignores_weight.toFixed(2):"–",tone:n?"bad":void 0,sub:n?e("the test catches the bug","test hatayı yakalar"):""})]})}const Z=8,Un=["<bos>","caption"," tr",`
`],ot=n=>n.replace("▁","␣").replace(`
`,"\\n");function nn(){const{data:n,error:e}=Ae("pg_slices");return{v:a.useMemo(()=>{if(!n)return null;const s=Un.map(i=>{const c=n.embed_named.findIndex(d=>d.text===i);return{piece:n.embed_named[c].piece,x:Array.from(tt(n.embed_named_rows,c).slice(0,Z),d=>Math.fround(d*45.25))}}),o=Array.from(tt(n.norms.input_layernorm,0).slice(0,Z));return{rows:s,w:o}},[n]),error:e}}const J=n=>Math.abs(n)>=100?n.toFixed(0):Math.abs(n)>=10?n.toFixed(1):n.toFixed(2),Ot=[["x","B, T, D",{en:"input",tr:"girdi"}],["x.pow(2)","B, T, D",{en:"elementwise square",tr:"eleman eleman kare"}],[".mean(-1, keepdim=True)","B, T, 1",{en:"mean over the last axis",tr:"son eksen boyunca ortalama"}],["torch.rsqrt(… + self.eps)","B, T, 1",{en:"1 / √(mean + ε)",tr:"1 / √(ortalama + ε)"}],["x * rsqrt  (broadcast)","B, T, D",{en:"the [T, 1] column is broadcast along D",tr:"[T, 1] sütunu D boyunca yayılıyor"}],["_norm(x)","B, T, D",{en:"every row now has RMS 1",tr:"her satırın RMS'i artık 1"}],["output * (1.0 + self.weight)","B, T, D",{en:"each dim with its own coefficient",tr:"her boyut kendi katsayısıyla"}]];function rr({step:n}){const e=rn(),r=ae(),{v:s,error:o}=nn(),[i,c]=Ut();if(!s)return t.jsx("div",{ref:i,children:t.jsx(it,{what:r("real embeddings","gerçek embedding'ler"),error:o})});const d=s.rows.length,p=s.rows.map(h=>Re(h.x,s.w)),x=4,u=Math.max(26,Math.min(56,(c-130-90)/(Z+2))),l=78,v=56,b=l+Z*(u+x)+34,S=h=>l+h*(u+x),C=h=>v+h*(u+x),F=v+d*(u+x)+50,_=[],D=h=>Math.max(...h.map(Math.abs));let f=1;for(let h=0;h<d;h++)for(let P=0;P<Z;P++){const A=p[h],U=A.x[P],B=A.sq[P],N=A.normed[P],L=A.out[P];let j=U,g=J(U),M=S(P),m=C(h),y=1,k;n===1&&(j=B,g=J(B)),(n===2||n===3)&&(M=b,y=0,j=B,g=""),n===4&&(y=0),n===5&&(j=N,g=J(N)),n>=6&&(j=L,g=J(L)),k=()=>{const R=t.jsxs("div",{className:"muted",style:{marginBottom:2},children:["[t=",h,' "',ot(s.rows[h].piece),'", d=',P,"]"]});return n===0?t.jsxs(t.Fragment,{children:[R,"x = ",t.jsx("b",{children:I(U,4)})]}):n===1?t.jsxs(t.Fragment,{children:[R,"x² = ",I(U,3),"² = ",t.jsx("b",{children:I(B,3)})]}):n===5?t.jsxs(t.Fragment,{children:[R,"x · rsqrt = ",I(U,3)," × ",I(A.rsqrt,4)," = ",t.jsx("b",{children:I(N,4)})]}):t.jsxs(t.Fragment,{children:[R,"normed · (1 + w[",P,"]) = ",I(N,3)," × ",I(A.scale[P],3)," = ",t.jsx("b",{children:I(L,4)})]})},_.push({id:`x-${h}-${P}`,x:M,y:m,w:u,h:u,v:j,text:u>34?g:"",opacity:y,tip:k,group:`r${h}`})}f=n===1?D(p.flatMap(h=>h.sq)):n===5?2.5:n>=6?D(p.flatMap(h=>h.out)):D(p.flatMap(h=>h.x));for(let h=0;h<d;h++){const P=p[h],A=n>=2&&n<=4,U=n>=3;_.push({id:`m-${h}`,x:b,y:C(h),w:u*1.5,h:u,opacity:A?1:0,fill:U?"var(--ax-T)":"var(--fg-muted)",text:A?J(U?P.rsqrt:P.meanSq):"",group:`r${h}`,z:3,tip:()=>U?t.jsxs(t.Fragment,{children:["rsqrt[t=",h,"] = 1/√(",I(P.meanSq,3)," + 1e-6) = ",t.jsx("b",{children:I(P.rsqrt,5)})]}):t.jsxs(t.Fragment,{children:["mean(x²) = (",P.sq.map(B=>J(B)).join(" + "),") / ",Z," = ",t.jsx("b",{children:I(P.meanSq,3)})]})});for(let B=0;B<Z;B++){const N=n===4;_.push({id:`b-${h}-${B}`,x:N?S(B):b,y:C(h),w:u,h:u,fill:"var(--ax-T)",opacity:N?.85:0,text:N&&u>34?J(P.rsqrt):"",group:`r${h}`,z:2,tip:()=>t.jsxs(t.Fragment,{children:[r("broadcast copy","yayılmış kopya"),": rsqrt[t=",h,"] = ",I(P.rsqrt,4),t.jsx("div",{className:"muted",children:r("no copy in memory, just a view","bellekte kopya yok, sadece görünüm")})]})})}}for(let h=0;h<Z;h++){const P=n>=6;_.push({id:`w-${h}`,x:S(h),y:P?v-u-14:v-u-60,w:u,h:u*.7,fill:"var(--ax-hd)",opacity:P?1:0,text:P&&u>34?J(1+s.w[h]):"",z:3,tip:()=>t.jsxs(t.Fragment,{children:["1 + w[",h,"] = 1 + ",I(s.w[h],4)," = ",t.jsx("b",{children:I(1+s.w[h],4)}),t.jsx("div",{className:"muted",children:r("real layer 0 input_layernorm","gerçek katman 0 input_layernorm")})]})})}const[E,z,O]=Ot[Math.min(n,Ot.length-1)],W=O[e],H=h=>{const P=n===5?p[h].normed:n>=6?p[h].out:n===1?p[h].sq:p[h].x;return Math.sqrt(P.reduce((A,U)=>A+U*U,0)/Z)},q=t.jsxs(t.Fragment,{children:[s.rows.map((h,P)=>t.jsx("div",{style:{position:"absolute",left:0,top:C(P),height:u,width:l-10,display:"flex",alignItems:"center",justifyContent:"flex-end",fontFamily:"var(--mono)",fontSize:12,color:"var(--ax-T)"},children:ot(h.piece)},P)),(n===0||n===5||n>=6)&&s.rows.map((h,P)=>t.jsxs("div",{style:{position:"absolute",left:b,top:C(P),height:u,display:"flex",alignItems:"center",fontSize:12,color:n===5?"var(--ok)":"var(--fg-muted)",fontFamily:"var(--mono)"},children:["rms = ",H(P).toFixed(2)]},"r"+P)),t.jsxs("div",{style:{position:"absolute",left:l,top:v+d*(u+x)+6,fontSize:12,color:"var(--ax-D)"},children:["D → (",r("2048 in reality, the first 8 here","gerçekte 2048, burada ilk 8"),")"]}),n>=2&&n<=5&&t.jsx("div",{style:{position:"absolute",left:b,top:v-22,fontSize:12,color:"var(--fg-muted)"},children:"[T, 1]"}),t.jsx("div",{style:{position:"absolute",left:0,top:v-22,width:l-10,textAlign:"right",fontSize:12,color:"var(--ax-T)"},children:"T ↓"})]});return t.jsxs("div",{ref:i,children:[t.jsxs("div",{style:{display:"flex",alignItems:"baseline",gap:14,flexWrap:"wrap",marginBottom:6},children:[t.jsx("code",{style:{fontSize:14},children:E}),t.jsx(sn,{s:z}),t.jsx("span",{className:"small muted",children:W})]}),t.jsx(Qt,{cells:_,width:c,height:F,max:f,labels:q}),t.jsx("div",{className:"small muted",children:r("Hover the cells to see how each number is computed.","Hücrelerin üstüne gel: her sayının nasıl hesaplandığı görünür.")})]})}function or(){const n=ae(),{v:e,error:r}=nn(),[s,o]=Ut(),[i,c]=a.useState({}),[d,p]=a.useState("f32"),[x,u]=a.useState("gemma");if(!e)return t.jsx("div",{ref:s,children:t.jsx(it,{what:n("real embeddings","gerçek embedding'ler"),error:r})});const l=i.x??e.rows.map(m=>m.x),v=i.w??e.w,b=l.length,S=l.map(m=>Re(m,v,1e-6,d,x)),C=l.map(m=>Re(m,v,1e-6,d,x==="gemma"?"llama":"gemma")),F=3,_=34,D=64,f=Math.max(20,Math.min(40,(o-D-4*_-3*F*Z)/(3*Z+1.4))),E=Z*(f+F)-F,z=D+E+_,O=z+f*1.4+_,W=O+E+_,H=34,q=m=>H+m*(f+F),h=H+b*(f+F)+8,P=H+(b*(f+F)-F)/2,A=[],U=Math.max(...l.flat().map(Math.abs),1),B=Math.max(...S.flatMap(m=>m.out).map(Math.abs),1);let N=0;for(let m=0;m<b;m++)for(let y=0;y<Z;y++){const k=S[m],R=k.out[y]!==C[m].out[y];R&&N++,A.push({id:`x${m}${y}`,x:D+y*(f+F),y:q(m),w:f,h:f,v:l[m][y]/U*2.5,text:f>30?J(l[m][y]):"",editable:!0,group:`r${m}`,tip:()=>t.jsxs(t.Fragment,{children:["x[",m,",",y,"] = ",t.jsx("b",{children:I(l[m][y],4)}),t.jsx("div",{className:"muted",children:n("drag up/down","yukarı/aşağı sürükle")})]})}),A.push({id:`o${m}${y}`,x:W+y*(f+F),y:q(m),w:f,h:f,v:k.out[y]/B*2.5,text:f>30?J(k.out[y]):"",ring:d==="bf16"&&R,group:`r${m}`,tip:()=>t.jsxs(t.Fragment,{children:[t.jsxs("div",{children:["x · rsqrt · (1+w) = ",I(k.x[y],3)," × ",I(k.rsqrt,4)," × ",I(k.scale[y],3)]}),t.jsxs("div",{children:["= ",t.jsx("b",{children:k.out[y]})," ",t.jsxs("span",{className:"muted",children:["(",d,", ",x," ",n("order","sırası"),")"]})]}),d==="bf16"&&t.jsxs("div",{className:"muted",children:[n("other order","öbür sıra"),": ",C[m].out[y]," ",R?n("· DIFFERENT","· FARKLI"):n("· same","· aynı")]})]})})}for(let m=0;m<b;m++)A.push({id:`r${m}`,x:z,y:q(m),w:f*1.4,h:f,fill:"var(--ax-T)",text:J(S[m].rsqrt),group:`r${m}`,tip:()=>t.jsxs(t.Fragment,{children:["rms(x) = ",I(Math.sqrt(S[m].meanSq),4)," → 1 / rms = ",t.jsx("b",{children:I(S[m].rsqrt,5)}),t.jsx("div",{className:"muted",children:n("every element of the row is multiplied by this","satırın bütün elemanları bununla çarpılıyor")})]})});for(let m=0;m<Z;m++)A.push({id:`w${m}`,x:O+m*(f+F),y:P-f/2,w:f,h:f,fill:"var(--ax-hd)",text:f>30?J(1+v[m]):"",editable:!0,tip:()=>t.jsxs(t.Fragment,{children:["1 + w[",m,"] = ",t.jsx("b",{children:I(1+v[m],4)}),t.jsx("div",{className:"muted",children:n(`column ${m} of every row is multiplied by this · drag`,`her satırın ${m}. sütunu bununla çarpılıyor · sürükle`)})]})});const L=(m,y)=>{if(m.startsWith("x")){const k=+m[1],R=+m.slice(2),G=l.map(T=>T.slice());G[k][R]=Math.round((G[k][R]+y*U/120)*100)/100,c({...i,x:G})}else if(m.startsWith("w")){const k=+m.slice(1),R=v.slice();R[k]=Math.round((R[k]+y/60)*100)/100,c({...i,w:R})}},j=(m,y,k,R,G="var(--fg)")=>t.jsxs("div",{style:{position:"absolute",left:m,width:y,top:0,textAlign:"center",fontSize:12.5,lineHeight:1.25,whiteSpace:"nowrap"},children:[t.jsx("b",{style:{color:G},children:k})," ",t.jsx("span",{className:"muted mono",style:{fontSize:11},children:R})]}),g=(m,y)=>t.jsx("div",{style:{position:"absolute",left:m,width:_,top:P-14,textAlign:"center",fontSize:24,lineHeight:"28px",color:"var(--fg-faint)",fontWeight:300},children:y}),M=t.jsxs(t.Fragment,{children:[j(D,E,t.jsxs(t.Fragment,{children:["x ",t.jsx("span",{style:{fontWeight:400},children:"↕"})]}),"[T, D]"),j(z-10,f*1.4+20,"1 / rms","[T, 1]","var(--ax-T)"),j(O,E,t.jsxs(t.Fragment,{children:["1 + w ",t.jsx("span",{style:{fontWeight:400},children:"↕"})]}),"[D]","var(--ax-hd)"),j(W,E,n("output","çıktı"),"[T, D]"),g(D+E,"×"),g(z+f*1.4,"×"),g(O+E,"="),l.map((m,y)=>t.jsx("div",{style:{position:"absolute",left:0,top:q(y),height:f,width:D-10,textAlign:"right",lineHeight:`${f}px`,fontFamily:"var(--mono)",fontSize:12,color:"var(--ax-T)"},children:ot(e.rows[y].piece)},y))]});return t.jsxs("div",{ref:s,children:[t.jsxs("div",{className:"row",style:{justifyContent:"space-between",alignItems:"flex-end",marginBottom:10},children:[t.jsxs("div",{className:"row",style:{gap:18},children:[t.jsx(je,{value:d,onChange:p,options:[["f32","float32"],["bf16","bfloat16"]]}),t.jsx(je,{value:x,onChange:u,options:[["gemma",n("Gemma order","Gemma sırası")],["llama",n("Llama order","Llama sırası")]]})]}),t.jsxs("div",{className:"row",style:{gap:4},children:[t.jsx("button",{className:"btn",onClick:()=>c({...i,x:l.map(m=>m.map(y=>y*10))}),children:"x × 10"}),t.jsx("button",{className:"btn",onClick:()=>c({...i,w:v.map(()=>0)}),children:"w = 0"}),t.jsx("button",{className:"btn",onClick:()=>c({}),children:n("real values","gerçek değerler")})]})]}),t.jsx(Qt,{cells:A,width:o,height:h,max:2.5,onDrag:L,labels:M}),t.jsxs("div",{className:"small muted",style:{marginTop:6},children:[t.jsx("b",{style:{color:"var(--fg)"},children:"↕"})," ",n("The marked blocks (x and 1 + w) can be dragged up and down.","işaretli bloklar (x ve 1 + w) yukarı-aşağı sürüklenebilir.")," ",d==="bf16"?t.jsxs(t.Fragment,{children:[t.jsx("span",{style:{display:"inline-block",width:12,height:12,borderRadius:2,outline:"2px solid var(--accent)",outlineOffset:1,verticalAlign:-1,margin:"0 5px 0 3px"}}),n(t.jsxs(t.Fragment,{children:["Ringed cells: where the Gemma and Llama orders give different results: ",t.jsxs("b",{style:{color:"var(--fg)"},children:[N," / ",b*Z]}),". The difference is 1 ulp, but an ",t.jsx("code",{children:"exact=True"})," test catches it."]}),t.jsxs(t.Fragment,{children:["Halkalı hücreler: Gemma ve Llama sırasının farklı sonuç verdiği yerler: ",t.jsxs("b",{style:{color:"var(--fg)"},children:[N," / ",b*Z]}),". Fark 1 ulp, ama ",t.jsx("code",{children:"exact=True"})," testi bunu yakalıyor."]}))]}):n(t.jsx(t.Fragment,{children:'Try "x × 10": rsqrt shrinks 10×, the output does not change at all. Switch to bfloat16 to see the difference between the two orders.'}),t.jsx(t.Fragment,{children:`"x × 10" dene: rsqrt 10 kat küçülür, çıktı hiç değişmez. bfloat16'ya geçince iki işlem sırasının farkı görünür.`}))]})]})}function sr(){const n=ae(),[e,r]=a.useState("rms"),[s,o]=a.useState(3),[i,c]=a.useState([.6,-.4,.2]),d=s;return t.jsxs("div",{children:[t.jsxs("div",{className:"row",style:{justifyContent:"space-between",alignItems:"flex-end",marginBottom:10,gap:16,flexWrap:"wrap"},children:[t.jsxs("div",{className:"row",style:{gap:18},children:[t.jsx(je,{value:e,onChange:r,options:[["rms","RMSNorm"],["w","RMSNorm · (1+w)"],["ln","LayerNorm"]]}),t.jsx(je,{value:s,onChange:o,options:[[3,n("3 dims","3 boyut")],[2,n("2 dims","2 boyut")]]})]}),e==="w"&&t.jsx("div",{className:"row small",style:{gap:16,alignItems:"center",flexWrap:"wrap"},children:i.slice(0,d).map((p,x)=>t.jsxs("label",{children:["1+w","₀₁₂"[x]," = ",t.jsx("b",{className:"mono",children:(1+p).toFixed(2)})," ",t.jsx("input",{type:"range",min:-1.5,max:1.5,step:.05,value:p,onChange:u=>c(i.map((l,v)=>v===x?+u.target.value:l))})]},x))})]}),s===3?t.jsx(Nn,{mode:e,w:i}):t.jsx(Hn,{mode:e,w:i})]})}const st=(n,e,r)=>n==="ln"?bn(e):Re(e,n==="w"?r.slice(0,e.length):e.map(()=>0)).out;function Nn({mode:n,w:e}){const r=ae(),s=a.useMemo(()=>{const b=lt(7);return Array.from({length:60},()=>[b.normal()*1.3+.5,b.normal()*1+.3,b.normal()*1.1+.2])},[]),[o,i]=a.useState([2.2,1.1,.6]),c=Math.sqrt(3),d=n==="ln"?"var(--ax-D)":n==="w"?"var(--ax-hd)":"var(--ax-T)",p=[];s.forEach((b,S)=>{const C=st(n,b,e);p.push({id:"i"+S,pos:b,size:[.07,.07,.07],color:ie("--fg-faint"),tip:()=>t.jsxs(t.Fragment,{children:[r("input","girdi")," (",b.map(F=>F.toFixed(2)).join(", "),")",t.jsxs("div",{className:"muted",children:["rms = ",Math.sqrt((b[0]**2+b[1]**2+b[2]**2)/3).toFixed(2)]})]})}),p.push({id:"o"+S,pos:C,size:[.11,.11,.11],color:ie(d.slice(4,-1)),tip:()=>t.jsxs(t.Fragment,{children:["norm = (",C.map(F=>F.toFixed(2)).join(", "),")",t.jsxs("div",{className:"muted",children:["|norm| = ",Math.hypot(...C).toFixed(3)]})]})})});const x=st(n,o,e),u=n==="w"?[Math.abs(1+e[0])*c,Math.abs(1+e[1])*c,Math.abs(1+e[2])*c]:[c,c,c],l=new w(1,1,1).normalize(),v=new at().setFromUnitVectors(new w(0,0,1),l);return t.jsxs("div",{children:[t.jsxs(qt,{view:{pos:[6.4,4.6,7.6],target:[0,.2,0]},height:500,minDist:3,maxDist:20,children:[t.jsx(Xt,{origin:[0,0,0],axes:[{dir:[1,0,0],len:3.2,color:"var(--fg-faint)",label:"x₀"},{dir:[0,1,0],len:3.2,color:"var(--fg-faint)",label:"x₁"},{dir:[0,0,1],len:3.2,color:"var(--fg-faint)",label:"x₂"}]}),n!=="ln"&&t.jsxs(Wn,{target:u,children:[t.jsxs("mesh",{children:[t.jsx("sphereGeometry",{args:[1,48,32]}),t.jsx("meshStandardMaterial",{color:ie(d.slice(4,-1)),transparent:!0,opacity:.1,depthWrite:!1})]}),t.jsxs("mesh",{children:[t.jsx("sphereGeometry",{args:[1.002,24,16]}),t.jsx("meshBasicMaterial",{color:ie(d.slice(4,-1)),wireframe:!0,transparent:!0,opacity:.16})]})]}),n==="ln"&&t.jsxs("group",{quaternion:v,children:[t.jsxs("mesh",{children:[t.jsx("circleGeometry",{args:[3.4,64]}),t.jsx("meshBasicMaterial",{color:ie("--ax-D"),transparent:!0,opacity:.07,side:Ue,depthWrite:!1})]}),t.jsxs("mesh",{children:[t.jsx("torusGeometry",{args:[c,.018,8,96]}),t.jsx("meshBasicMaterial",{color:ie("--ax-D")})]})]}),t.jsx(Vt,{items:p,tween:{duration:.9,stagger:.004}}),t.jsx($n,{value:o,onChange:i}),t.jsx(In,{a:o,b:x}),t.jsxs("mesh",{position:x,children:[t.jsx("sphereGeometry",{args:[.11,20,20]}),t.jsx("meshBasicMaterial",{color:ie("--accent")})]}),t.jsx(Kt,{position:[x[0],x[1]+.3,x[2]],color:"var(--accent)",weight:650,children:"norm(x)"})]}),t.jsxs("div",{className:"small muted",style:{marginTop:8},children:[r("Drag the big blue point by its arrows.","Büyük mavi noktayı oklarıyla sürükle.")," x = (",o.map(b=>b.toFixed(2)).join(", "),"), rms(x) = ",Math.sqrt((o[0]**2+o[1]**2+o[2]**2)/3).toFixed(3)," → norm(x) = (",x.map(b=>b.toFixed(2)).join(", "),")."," ",n==="rms"&&r("All points land on the sphere of radius √3: direction is kept, magnitude is dropped. At d = 2048 the radius is √2048 ≈ 45.25.","Bütün noktalar √3 yarıçaplı küreye iniyor: yön korunuyor, büyüklük atılıyor. d = 2048'de küre yarıçapı √2048 ≈ 45.25."),n==="w"&&r("(1+w) scales each axis separately: the sphere becomes an ellipsoid. Pull a coefficient below 0 and that axis flips.","(1+w) her ekseni ayrı ölçekliyor: küre elipsoide dönüşüyor. Bir katsayıyı 0'ın altına çek, o eksen ters döner."),n==="ln"&&r("LayerNorm subtracts the mean first: the points drop onto the plane x₀ + x₁ + x₂ = 0. Then it scales them onto the circle of radius √3 in that plane. The shared shift of all dims, the (1, 1, 1) direction, is erased. RMSNorm keeps it.","LayerNorm önce ortalamayı çıkarıyor: noktalar x₀ + x₁ + x₂ = 0 düzlemine iniyor. Sonra ölçekliyor: düzlemdeki √3 yarıçaplı çembere. Bütün boyutların ortak kayması (1, 1, 1) yönü silindi. RMSNorm bu bilgiyi koruyor.")]})]})}function Wn({target:n,children:e}){const r=a.useRef(null),s=a.useMemo(()=>new w,[]);return Gt((o,i)=>{r.current&&hn.damp3(r.current.scale,s.set(...n),.3,i)}),t.jsx("group",{ref:r,scale:n,children:e})}function In({a:n,b:e}){const r=a.useMemo(()=>new pn,[]);return r.setFromPoints([new w(...n),new w(...e)]),t.jsx("lineSegments",{geometry:r,children:t.jsx("lineBasicMaterial",{color:ie("--accent")})})}function $n({value:n,onChange:e}){const r=le(i=>i.controls),s=a.useMemo(()=>new $().makeTranslation(n[0],n[1],n[2]),[]),o=a.useMemo(()=>new w,[]);return t.jsx(On,{matrix:s,autoTransform:!0,disableRotations:!0,disableScaling:!0,disableSliders:!0,scale:.9,depthTest:!1,lineWidth:2.5,onDragStart:()=>r&&(r.enabled=!1),onDragEnd:()=>r&&(r.enabled=!0),onDrag:i=>{o.setFromMatrixPosition(i),e([+o.x.toFixed(2),+o.y.toFixed(2),+o.z.toFixed(2)])},children:t.jsxs("mesh",{children:[t.jsx("sphereGeometry",{args:[.13,24,24]}),t.jsx("meshStandardMaterial",{color:ie("--accent")})]})})}function Hn({mode:n,w:e}){const r=ae(),[s,o]=e,i=xn([2.2,.8],{color:"var(--accent)"}),c=a.useMemo(()=>{const l=lt(4);return Array.from({length:36},()=>[l.normal()*1.4+.4,l.normal()*1.1+.3])},[]),d=Math.SQRT2,p=l=>{const v=st(n,l,e);return[v[0],v[1]]},x=p(i.point),u=Math.sqrt((i.x*i.x+i.y*i.y)/2);return t.jsxs("div",{children:[t.jsx("div",{style:{maxWidth:760,margin:"0 auto"},children:t.jsxs(gn,{height:470,viewBox:{x:[-4,4],y:[-2.6,2.6]},preserveAspectRatio:"contain",children:[t.jsx(yn.Cartesian,{subdivisions:!1,xAxis:{lines:1,labels:l=>l%2===0?l:""},yAxis:{lines:1,labels:l=>l%2===0?l:""}}),n==="rms"&&t.jsx(vn,{center:[0,0],radius:d,color:"var(--ax-T)",fillOpacity:.04,strokeStyle:"dashed"}),n==="w"&&t.jsx(wn,{center:[0,0],radius:[d*Math.abs(1+s),d*Math.abs(1+o)],color:"var(--ax-hd)",fillOpacity:.04,strokeStyle:"dashed"}),n==="ln"&&t.jsx(Ne.ThroughPoints,{point1:[-2,2],point2:[2,-2],color:"var(--ax-D)",style:"dashed"}),c.map((l,v)=>{const b=p(l);return t.jsxs("g",{children:[t.jsx(Ne.Segment,{point1:l,point2:b,color:"var(--fg-faint)",opacity:.25,weight:1}),t.jsx(We,{x:l[0],y:l[1],color:"var(--fg-faint)",opacity:.5}),t.jsx(We,{x:b[0],y:b[1],color:n==="ln"?"var(--ax-D)":n==="w"?"var(--ax-hd)":"var(--ax-T)",opacity:.7})]},v)}),t.jsx(Ne.Segment,{point1:i.point,point2:x,color:"var(--accent)",weight:2}),t.jsx(We,{x:x[0],y:x[1],color:"var(--accent)"}),t.jsx(ht,{x:x[0],y:x[1],attach:x[1]>=0?"n":"s",attachDistance:18,size:13,color:"var(--accent)",children:"norm(x)"}),i.element,t.jsx(ht,{x:i.x,y:i.y,attach:"ne",attachDistance:14,size:13,color:"var(--accent)",children:"x"})]})}),t.jsxs("div",{className:"small muted",style:{marginTop:8},children:[r("Drag the point x.","x noktasını sürükle.")," x = (",i.x.toFixed(2),", ",i.y.toFixed(2),"), rms(x) = ",u.toFixed(3)," → norm(x) = (",x[0].toFixed(3),", ",x[1].toFixed(3),")."," ",n==="rms"&&r("Wherever you drag it, its image stays on the circle of radius √2: direction is kept, magnitude is dropped.","Nereye sürüklersen sürükle, görüntüsü √2 yarıçaplı çemberden ayrılmıyor: yön korunuyor, büyüklük atılıyor."),n==="w"&&r("(1+w) scales each axis separately: the circle becomes an ellipse. Pull a coefficient below 0 and that axis flips.","(1+w) her ekseni ayrı ölçekliyor: çember elipse dönüşüyor. Bir katsayıyı 0'ın altına çek, o eksen ters döner."),n==="ln"&&r("At d = 2 only two points survive LayerNorm: (1, −1) and (−1, 1). That is misleadingly harsh; switch to 3 dims to see what really happens.","d = 2'de LayerNorm'dan geriye sadece iki nokta kalıyor: (1, −1) ve (−1, 1). Bu yanıltıcı derecede sert. 3 boyuta geçince ne olduğunu daha iyi görürsün.")]})]})}const ce=2,se=3,Q=6;function ir({step:n}){const e=ae(),r=a.useMemo(()=>{const u=lt(9);return Array.from({length:ce*se*Q},()=>u.normal())},[]),s=.5,o=.56,i=.1,c=(u,l,v)=>[(v-Q/2)*o,(se/2-l)*o,(u-ce/2)*o*1.7],d=[],p=u=>u>=0?`hsl(18, ${40+Math.min(1,u)*40}%, ${70-Math.min(1,u)*22}%)`:`hsl(232, ${40+Math.min(1,-u)*40}%, ${72-Math.min(1,-u)*22}%)`;for(let u=0;u<ce;u++)for(let l=0;l<se;l++){const v=Array.from({length:Q},(S,C)=>r[(u*se+l)*Q+C]),b=v.reduce((S,C)=>S+C*C,0)/Q;for(let S=0;S<Q;S++){const C=v[S],F=n===1;d.push({id:`x${u}${l}${S}`,pos:F?c(u,l,Q+.8):c(u,l,S),color:p(C),hidden:F,dim:n===2,tip:()=>t.jsxs(t.Fragment,{children:["x[b=",u,", t=",l,", d=",S,"] = ",t.jsx("b",{children:C.toFixed(3)})]})})}d.push({id:`m${u}${l}`,pos:c(u,l,Q+.8),color:"#2a78d6",hidden:n===0,tip:()=>t.jsxs(t.Fragment,{children:["mean(x²)[b=",u,", t=",l,"] = ",t.jsx("b",{children:b.toFixed(3)}),t.jsxs("div",{className:"muted",children:[e("shape","şekil")," [B, T, 1]"]})]})});for(let S=0;S<Q;S++)d.push({id:`c${u}${l}${S}`,pos:n===2?[c(u,l,S)[0],c(u,l,S)[1]+0,c(u,l,S)[2]+.3]:c(u,l,Q+.8),color:"#2a78d6",hidden:n!==2,size:[s*.7,s*.7,s*.25],tip:()=>t.jsxs(t.Fragment,{children:[e("broadcast","yayılmış"),": rms[b=",u,", t=",l,"] (",e("the same number, D times","aynı sayı, D kez"),")"]})})}const x=[{pos:[2,2.3,4.6],target:[.1,.1,-.4]},{pos:[3.8,2,4],target:[1.3,.1,-.4]},{pos:[1.1,2.1,4.9],target:[.3,.1,-.4]}];return t.jsxs(qt,{view:x[Math.min(n,2)],height:420,children:[t.jsx(Vt,{items:d,cell:s,tween:{duration:.8,stagger:.008}}),t.jsx(Xt,{origin:[(n===1?Q+.8-Q/2:-Q/2)*o-s/2-i,se/2*o+s/2+i,(ce/2-1)*o*1.7+s/2+i],axes:[{dir:[1,0,0],len:(n===1?1:Q)*o+i,ticks:n===1?1:Q,color:"var(--ax-D)",label:n===1?"D = 1":`D = ${Q}`,labelOffset:[-.45,.3,0]},{dir:[0,-1,0],len:se*o+i,ticks:se,color:"var(--ax-T)",label:`T = ${se}`},{dir:[0,0,-1],len:(ce-1)*o*1.7+s+i,ticks:ce,color:"var(--ax-B)",label:`B = ${ce}`}]}),n>=1&&t.jsx(Kt,{position:[(Q/2+.8)*o,se/2*o+.75,.4],color:"var(--ax-T)",weight:650,children:n===1?"mean(-1, keepdim=True) → [B, T, 1]":"broadcast: [B, T, 1] → [B, T, D]"})]})}export{ir as K,sr as N,tr as O,rr as R,nr as Z,or as a,er as b};
