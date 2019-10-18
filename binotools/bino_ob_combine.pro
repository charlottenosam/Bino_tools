;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; input parameters:
;    inpdir -- a string array containing a list of input paths (DO NOT FORGET THE TRALING "/")
;    outdir -- a string defining the ouput path (DO NOT FORGET THE TRALING "/")
; optional input parameters:
;    fdata -- a filename for the flux (default = 'obj_abs_slits_lin.fits')
;    efdata -- a filename for the flux (default = 'obj_abs_err_slits_lin.fits')
;    knorm -- an array with normalization factors for individual observing blocks (defaults to all 1s)
;    /gzinp -- a flag should be set to read gzipped input files
; example:
;    bino_ob_combine,['ComaA_384_20180604/','ComaA_384_20180605/'],'ComaA_384_combined/',/gz
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;

pro bino_ob_combine,inpdir, outdir, $
    fdata=fdata, efdata=efdata, gzinp=gzinp,$
    knorm=knorm

n_ob=n_elements(inpdir)
multifile=0

if(n_elements(fdata) eq 0) then fdata='obj_abs_slits_lin.fits'
if(n_elements(efdata) eq 0) then efdata='obj_abs_err_slits_lin.fits'

if(n_elements(fdata) gt 1) then begin
   multifile=1
   if n_elements(fdata) ne n_ob then begin
     print, 'Flux filenames specified as an array, but count does not match count of input directories. Please check.'
     return
   endif
endif
if(n_elements(efdata) le 1) and (multifile eq 1) then begin
  print, 'Flux filenames specified as an array, please also specify error filenames as an array'
  return
endif

mkdir,outdir

suff=(keyword_set(gzinp))? '.gz' : ''
f1data=inpdir[0]+fdata[0]+suff
ef1data=inpdir[0]+efdata[0]+suff

fits_open,f1data, fcb
nslits=fcb.nextend
fits_close,fcb

f_odata=outdir+fdata[0]
ef_odata=outdir+efdata[0]

if(n_elements(knorm) ne n_ob) then knorm=dblarr(n_ob)+1d ;; normalization of exposures

h0_1=headfits(f1data)
exptime=sxpar(h0_1,'EXPTIME')
for i=1,n_ob-1 do begin
    if multifile then h0_c=headfits(inpdir[i]+fdata[i]+suff) else h0_c=headfits(inpdir[i]+fdata+suff)
    exptime+=sxpar(h0_c,'EXPTIME')
endfor
h0_o=h0_1

eh0_1=headfits(ef1data)
eh0_o=h0_1

sxaddpar,h0_o,'EXPTIME',exptime
sxaddpar,eh0_o,'EXPTIME',exptime
writefits,f_odata,0,h0_o
writefits,ef_odata,0,eh0_o

for i=0,nslits-1 do begin
    r1=mrdfits(f1data,i+1,h1,/silent)
    er1=mrdfits(ef1data,i+1,eh1,/silent)
    nx1=sxpar(h1,'NAXIS1')
    ny1=sxpar(h1,'NAXIS2')
    r_cube=dblarr(nx1,ny1,n_ob)+!values.d_nan
    er_cube=dblarr(nx1,ny1,n_ob)+!values.d_nan
    exparr=dblarr(nx1,ny1,n_ob)
    r_cube[*,*,0]=r1
    er_cube[*,*,0]=er1
    exparr[*,*,0]=knorm[0]

    print,'Slit',i+1,'/',nslits,' OB=',0,' ny    =',ny1
    for k=1,n_ob-1 do begin
        if multifile eq 1 then begin
          r_cur=mrdfits(inpdir[k]+fdata[k]+suff,i+1,h_cur,/silent)
          er_cur=mrdfits(inpdir[k]+efdata[k]+suff,i+1,eh_cur,/silent)
        endif else begin
          r_cur=mrdfits(inpdir[k]+fdata+suff,i+1,h_cur,/silent)
          er_cur=mrdfits(inpdir[k]+efdata+suff,i+1,eh_cur,/silent)
        endelse
        ny_cur=sxpar(h_cur,'NAXIS2')
        dy = (ny1-ny_cur)/2.0

        print,'Slit',i+1,'/',nslits,' OB=',k,' ny_cur=',ny_cur,' dy=',dy
        if(dy gt 0) then begin
            r_cube[*,dy:dy+ny_cur-1,k]=r_cur
            er_cube[*,dy:dy+ny_cur-1,k]=er_cur
            exparr[*,dy:dy+ny_cur-1,k]=knorm[k]
        endif else begin
            r_cube[*,*,k]=r_cur[*,-dy:-dy+ny1-1]
            er_cube[*,*,k]=er_cur[*,-dy:-dy+ny1-1]
            exparr[*,*,k]=knorm[k]
        endelse
    endfor

    exparr*=double((finite(r_cube) eq 1))
    r_total=total(r_cube,3,/nan)/total(exparr,3)
    er_total=sqrt(total(er_cube^2,3,/nan))/total(exparr,3)

    h_out=h1
    sxaddpar,h_out,'EXPTIME',exptime
    eh_out=eh1
    sxaddpar,eh_out,'EXPTIME',exptime
    mwrfits,float(r_total),f_odata,h_out,/silent
    mwrfits,float(er_total),ef_odata,eh_out,/silent
endfor

end
