	.arch armv8.5-a
	.build_version macos,  16, 0
	.text
Ltext0:
	.file 1 "/Users/albertstarfield/Documents/misc/AI/project-zephyrine/systemCore/engineMain/_excludefromRuntime_reverseEngineeringAssets/ada_courseRE/hello_sekai_spark/src/hello_sekai_spark.adb"
	.align	2
_hello_sekai_spark___finalizer.0:
LFB2:
	stp	x29, x30, [sp, -16]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	x0, x16
	bl	_system__secondary_stack__ss_release
	ldp	x29, x30, [sp], 16
LCFI2:
	ret
LFE2:
	.const
	.align	3
lC0:
	.ascii "hello_sekai_spark.adb"
	.space 1
	.text
	.align	2
	.globl __ada_hello_sekai_spark
__ada_hello_sekai_spark:
LFB1:
	.loc 1 7 1
	stp	x29, x30, [sp, -64]!
LCFI3:
	mov	x29, sp
LCFI4:
LEHB0:
LEHE0:
	str	x19, [sp, 16]
LCFI5:
	.loc 1 7 1 discriminator 1
	add	x0, x29, 64
	str	x0, [x29, 56]
	add	x8, x29, 32
LEHB1:
	bl	_system__secondary_stack__ss_mark
LVL0:
	.loc 1 11 54
	bl	_message_generator__get_message
LVL1:
	.loc 1 11 54 is_stmt 0 discriminator 2
	ldr	w2, [x1, 4]
LVL2:
	.loc 1 11 54 discriminator 14
	and	w2, w2, w2, asr #31
LVL3:
	ldr	w3, [x1]
	cmp	w3, w2
	ble	L10
	.loc 1 14 15 is_stmt 1
	bl	_ada__text_io__put_line__2
LVL4:
	b	L11
LVL5:
L10:
	.loc 1 11 54 discriminator 15
	mov	w1, 11
LVL6:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	___gnat_rcheck_CE_Range_Check
LVL7:
LEHE1:
L6:
	.loc 1 15 0 discriminator 5
	mov	x19, x0
	add	x16, x29, 32
LEHB2:
	bl	_hello_sekai_spark___finalizer.0
LVL8:
	mov	x0, x19
	bl	__Unwind_Resume
LVL9:
L11:
	.loc 1 15 0 is_stmt 0
	add	x16, x29, 32
	bl	_hello_sekai_spark___finalizer.0
LVL10:
	.loc 1 15 5 is_stmt 1
	ldr	x19, [sp, 16]
LEHE2:
	ldp	x29, x30, [sp], 64
LCFI6:
	ret
LFE1:
	.section __TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table0:
LLSDA1:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 LLSDACSE1-LLSDACSB1
LLSDACSB1:
	.uleb128 LEHB0-LFB1
	.uleb128 LEHE0-LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 LEHB1-LFB1
	.uleb128 LEHE1-LEHB1
	.uleb128 L6-LFB1
	.uleb128 0
	.uleb128 LEHB2-LFB1
	.uleb128 LEHE2-LEHB2
	.uleb128 0
	.uleb128 0
LLSDACSE1:
	.text
	.section __DWARF,__debug_frame,regular,debug
Lsection__debug_frame:
Lframe0:
	.set L$set$0,LECIE0-LSCIE0
	.long L$set$0
LSCIE0:
	.long	0xffffffff
	.byte	0x3
	.ascii "\0"
	.uleb128 0x1
	.sleb128 -8
	.uleb128 0x1e
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LECIE0:
LSFDE0:
	.set L$set$1,LEFDE0-LASFDE0
	.long L$set$1
LASFDE0:
	.set L$set$2,Lframe0-Lsection__debug_frame
	.long L$set$2
	.quad	LFB2
	.set L$set$3,LFE2-LFB2
	.quad L$set$3
	.byte	0x4
	.set L$set$4,LCFI0-LFB2
	.long L$set$4
	.byte	0xe
	.uleb128 0x10
	.byte	0x9d
	.uleb128 0x2
	.byte	0x9e
	.uleb128 0x1
	.byte	0x4
	.set L$set$5,LCFI1-LCFI0
	.long L$set$5
	.byte	0xd
	.uleb128 0x1d
	.byte	0x4
	.set L$set$6,LCFI2-LCFI1
	.long L$set$6
	.byte	0xde
	.byte	0xdd
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LEFDE0:
LSFDE2:
	.set L$set$7,LEFDE2-LASFDE2
	.long L$set$7
LASFDE2:
	.set L$set$8,Lframe0-Lsection__debug_frame
	.long L$set$8
	.quad	LFB1
	.set L$set$9,LFE1-LFB1
	.quad L$set$9
	.byte	0x4
	.set L$set$10,LCFI3-LFB1
	.long L$set$10
	.byte	0xe
	.uleb128 0x40
	.byte	0x9d
	.uleb128 0x8
	.byte	0x9e
	.uleb128 0x7
	.byte	0x4
	.set L$set$11,LCFI4-LCFI3
	.long L$set$11
	.byte	0xd
	.uleb128 0x1d
	.byte	0x4
	.set L$set$12,LCFI5-LCFI4
	.long L$set$12
	.byte	0x93
	.uleb128 0x6
	.byte	0x4
	.set L$set$13,LCFI6-LCFI5
	.long L$set$13
	.byte	0xde
	.byte	0xdd
	.byte	0xd3
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LEFDE2:
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$14,LECIE1-LSCIE1
	.long L$set$14
LSCIE1:
	.long	0
	.byte	0x3
	.ascii "zPLR\0"
	.uleb128 0x1
	.sleb128 -8
	.uleb128 0x1e
	.uleb128 0x7
	.byte	0x9b
L_got_pcr0:
	.long	___gnat_personality_v0@GOT-L_got_pcr0
	.byte	0x10
	.byte	0x10
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LECIE1:
LSFDE5:
	.set L$set$15,LEFDE5-LASFDE5
	.long L$set$15
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB2-.
	.set L$set$16,LFE2-LFB2
	.quad L$set$16
	.uleb128 0x8
	.quad	0
	.byte	0x4
	.set L$set$17,LCFI0-LFB2
	.long L$set$17
	.byte	0xe
	.uleb128 0x10
	.byte	0x9d
	.uleb128 0x2
	.byte	0x9e
	.uleb128 0x1
	.byte	0x4
	.set L$set$18,LCFI1-LCFI0
	.long L$set$18
	.byte	0xd
	.uleb128 0x1d
	.byte	0x4
	.set L$set$19,LCFI2-LCFI1
	.long L$set$19
	.byte	0xde
	.byte	0xdd
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LEFDE5:
LSFDE7:
	.set L$set$20,LEFDE7-LASFDE7
	.long L$set$20
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB1-.
	.set L$set$21,LFE1-LFB1
	.quad L$set$21
	.uleb128 0x8
	.quad	LLSDA1-.
	.byte	0x4
	.set L$set$22,LCFI3-LFB1
	.long L$set$22
	.byte	0xe
	.uleb128 0x40
	.byte	0x9d
	.uleb128 0x8
	.byte	0x9e
	.uleb128 0x7
	.byte	0x4
	.set L$set$23,LCFI4-LCFI3
	.long L$set$23
	.byte	0xd
	.uleb128 0x1d
	.byte	0x4
	.set L$set$24,LCFI5-LCFI4
	.long L$set$24
	.byte	0x93
	.uleb128 0x6
	.byte	0x4
	.set L$set$25,LCFI6-LCFI5
	.long L$set$25
	.byte	0xde
	.byte	0xdd
	.byte	0xd3
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LEFDE7:
	.text
Letext0:
	.section __DWARF,__debug_info,regular,debug
Lsection__debug_info:
Ldebug_info0:
	.long	0x4f9
	.short	0x4
	.set L$set$26,Ldebug_abbrev0-Lsection__debug_abbrev
	.long L$set$26
	.byte	0x8
	.uleb128 0x1
	.ascii "GNU Ada 15.0.1 20250418 (prerelease) -Og -gnatA -ffunction-sections -fdata-sections -g -gnatwa -gnatw.X -gnatVa -gnaty3 -gnatya -gnatyA -gnatyB -gnatyb -gnatyc -gnaty-d -gnatye -gnatyf -gnatyh -gnatyi -gnatyI -gnatyk -gnatyl -gnatym -gnatyn -gnatyO -gnatyp -gnatyr -gnatyS -gnatyt -gnatyu -gnatyx -gnatW8 -gnatR2js -gnatws -gnatis -gnatec=/private/var/folders/vj/2td27x090rqc1ln_jr_6v83m0000gn/T/GPR.27480/GNAT-TEMP-000003.TMP -gnatem=/private/var/folders/vj/2td27x090rqc1ln_jr_6v83m0000gn/T/GPR.27480/GNAT-TEMP-000004.TMP -mmacosx-version-min=16.0.0 -mcpu=apple-m1 -mlittle-endian -mabi=lp64 -fPIC\0"
	.byte	0xd
	.ascii "/Users/albertstarfield/Documents/misc/AI/project-zephyrine/systemCore/engineMain/_excludefromRuntime_reverseEngineeringAssets/ada_courseRE/hello_sekai_spark/src/hello_sekai_spark.adb\0"
	.ascii "/Users/albertstarfield/Documents/misc/AI/project-zephyrine/systemCore/engineMain/_excludefromRuntime_reverseEngineeringAssets/ada_courseRE/hello_sekai_spark/obj/development/gnatprove/data_representation\0"
	.quad	Ltext0
	.set L$set$27,Letext0-Ltext0
	.quad L$set$27
	.set L$set$28,Ldebug_line0-Lsection__debug_line
	.long L$set$28
	.uleb128 0x2
	.byte	0x8
	.byte	0x5
	.ascii "system__parameters__Tsize_typeB\0"
	.uleb128 0x3
	.byte	0x1
	.byte	0x2
	.ascii "boolean\0"
	.uleb128 0x3
	.byte	0x1
	.byte	0x7
	.ascii "system__storage_elements__storage_element\0"
	.uleb128 0x4
	.ascii "hello_sekai_spark\0"
	.byte	0x1
	.byte	0x7
	.byte	0x1
	.ascii "_ada_hello_sekai_spark\0"
	.quad	LFB1
	.set L$set$29,LFE1-LFB1
	.quad L$set$29
	.uleb128 0x1
	.byte	0x9c
	.long	0x4e4
	.uleb128 0x5
	.ascii "B2b\0"
	.long	0x4e4
	.set L$set$30,LLST0-Lsection__debug_loc
	.long L$set$30
	.uleb128 0x5
	.ascii "B6b\0"
	.long	0x4e4
	.set L$set$31,LLST1-Lsection__debug_loc
	.long L$set$31
	.uleb128 0x6
	.long	0x4ef
	.long	0x4c9
	.uleb128 0x7
	.long	0x4e4
	.long	0x498
	.long	0x4a5
	.byte	0
	.uleb128 0x8
	.ascii "the_message\0"
	.byte	0x1
	.byte	0xb
	.byte	0x4
	.long	0x4dd
	.uleb128 0x9
	.byte	0x8
	.long	0x4b2
	.byte	0
	.uleb128 0x3
	.byte	0x4
	.byte	0x5
	.ascii "integer\0"
	.uleb128 0x3
	.byte	0x1
	.byte	0x8
	.ascii "character\0"
	.byte	0
	.section __DWARF,__debug_abbrev,regular,debug
Lsection__debug_abbrev:
Ldebug_abbrev0:
	.uleb128 0x1
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0x8
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x1b
	.uleb128 0x8
	.uleb128 0x2134
	.uleb128 0x19
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x2
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x34
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x3
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.byte	0
	.byte	0
	.uleb128 0x4
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x6e
	.uleb128 0x8
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x5
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x2
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x6
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x7
	.uleb128 0x21
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x22
	.uleb128 0x13
	.uleb128 0x2f
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x8
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x9
	.uleb128 0x10
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.byte	0
	.section __DWARF,__debug_loc,regular,debug
Lsection__debug_loc:
Ldebug_loc0:
LLST0:
	.set L$set$32,LVL1-Ltext0
	.quad L$set$32
	.set L$set$33,LVL4-1-Ltext0
	.quad L$set$33
	.short	0x2
	.byte	0x71
	.sleb128 0
	.set L$set$34,LVL5-Ltext0
	.quad L$set$34
	.set L$set$35,LVL6-Ltext0
	.quad L$set$35
	.short	0x2
	.byte	0x71
	.sleb128 0
	.set L$set$36,LVL6-Ltext0
	.quad L$set$36
	.set L$set$37,LVL7-1-Ltext0
	.quad L$set$37
	.short	0x1
	.byte	0x53
	.quad	0
	.quad	0
LLST1:
	.set L$set$38,LVL2-Ltext0
	.quad L$set$38
	.set L$set$39,LVL3-Ltext0
	.quad L$set$39
	.short	0x1
	.byte	0x52
	.set L$set$40,LVL3-Ltext0
	.quad L$set$40
	.set L$set$41,LVL4-1-Ltext0
	.quad L$set$41
	.short	0x2
	.byte	0x71
	.sleb128 4
	.set L$set$42,LVL5-Ltext0
	.quad L$set$42
	.set L$set$43,LVL6-Ltext0
	.quad L$set$43
	.short	0x2
	.byte	0x71
	.sleb128 4
	.quad	0
	.quad	0
	.section __DWARF,__debug_pubnames,regular,debug
Lsection__debug_pubnames:
	.long	0x24
	.short	0x2
	.set L$set$44,Ldebug_info0-Lsection__debug_info
	.long L$set$44
	.long	0x4fd
	.long	0x455
	.ascii "hello_sekai_spark\0"
	.long	0
	.section __DWARF,__debug_pubtypes,regular,debug
Lsection__debug_pubtypes:
	.long	0x62
	.short	0x2
	.set L$set$45,Ldebug_info0-Lsection__debug_info
	.long L$set$45
	.long	0x4fd
	.long	0x41d
	.ascii "boolean\0"
	.long	0x428
	.ascii "system__storage_elements__storage_element\0"
	.long	0x4e4
	.ascii "integer\0"
	.long	0x4ef
	.ascii "character\0"
	.long	0
	.section __DWARF,__debug_aranges,regular,debug
Lsection__debug_aranges:
	.long	0x2c
	.short	0x2
	.set L$set$46,Ldebug_info0-Lsection__debug_info
	.long L$set$46
	.byte	0x8
	.byte	0
	.short	0
	.short	0
	.quad	Ltext0
	.set L$set$47,Letext0-Ltext0
	.quad L$set$47
	.quad	0
	.quad	0
	.section __DWARF,__debug_line,regular,debug
Lsection__debug_line:
Ldebug_line0:
	.section __DWARF,__debug_str,regular,debug
Lsection__debug_str:
	.ident	"GCC: (GNU) 15.0.1 20250418 (prerelease)"
	.subsections_via_symbols
